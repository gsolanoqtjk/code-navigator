#!/usr/bin/env python3
"""Code Mapper - Generates a structural map/graph of a codebase for token-efficient navigation.

This module creates a lightweight index of functions, classes, methods, and their
relationships within a codebase. The generated index can be used for quick symbol
lookup without reading entire files.

Example:
    Command line usage:
        $ codenav scan /path/to/project -o .codenav.json

    Python API usage:
        >>> mapper = CodeNavigator('/path/to/project')
        >>> code_map = mapper.scan()
        >>> print(code_map['stats'])
        {'files_processed': 142, 'symbols_found': 1847, 'errors': 0}

Attributes:
    LANGUAGE_EXTENSIONS: Dict mapping language names to file extensions.
    DEFAULT_IGNORE_PATTERNS: List of patterns to ignore when scanning.
"""

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .colors import get_colors

__version__ = "1.2.0"

# Supported languages and their extensions
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "go": [".go"],
    "rust": [".rs"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
    "dart": [".dart"],
}

DEFAULT_IGNORE_PATTERNS = [
    # Build artifacts and dependencies
    "node_modules",
    "__pycache__",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".nyc_output",
    "*.min.js",
    "*.bundle.js",
    ".tox",
    "eggs",
    "*.egg-info",
    ".pytest_cache",
    "vendor",
    "target",
    "bin",
    "obj",
    # Flutter/Dart build artifacts and generated files
    ".dart_tool",
    ".flutter-plugins",
    ".flutter-plugins-dependencies",
    "*.g.dart",
    "*.freezed.dart",
    "*.gr.dart",
    # Version control
    ".git",
    ".svn",
    ".hg",
    # IDE settings
    ".idea",
    ".vscode",
    # Environment files - ALL variants (SECURITY: prevents exposure of secrets)
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    ".env.production*",
    ".env.development*",
    ".envrc",
    "*.env",
    # Credentials and secrets (SECURITY)
    "secrets*",
    "*secret*",
    "*secrets*",
    "*credential*",
    "*credentials*",
    ".aws",
    ".gcp",
    ".ssh",
    ".gnupg",
    # Keys and certificates (SECURITY)
    "*.pem",
    "*.key",
    "*.p8",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*.crt",
    "*.cer",
    # Config files with potential secrets (SECURITY)
    ".npmrc",
    ".pypirc",
    ".netrc",
    "config/database.yml",
    "config/secrets.yml",
    # API keys and tokens
    "*apikey*",
    "*api_key*",
    "*token*",
]


@dataclass
class Symbol:
    """Represents a code symbol (function, class, method, etc.).

    Attributes:
        name: The symbol's name (e.g., 'process_payment').
        type: The symbol type ('function', 'class', 'method', 'variable', 'import').
        file_path: Relative path to the file containing the symbol.
        line_start: Starting line number (1-indexed).
        line_end: Ending line number (1-indexed, inclusive).
        signature: Function/class signature (e.g., 'def foo(x: int) -> str').
        docstring: First few lines of docstring, if present.
        parent: For methods, the containing class name.
        dependencies: List of symbols this symbol calls/uses.
        decorators: List of decorator names applied to this symbol.

    Example:
        >>> symbol = Symbol(
        ...     name='process_payment',
        ...     type='function',
        ...     file_path='src/billing.py',
        ...     line_start=45,
        ...     line_end=89,
        ...     signature='def process_payment(user_id: int, amount: Decimal)'
        ... )
    """

    name: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    signature: str | None = None
    docstring: str | None = None
    parent: str | None = None
    dependencies: list[str] = None
    decorators: list[str] = None
    truncated: bool = False  # True if symbol exceeded max line limit during analysis

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.decorators is None:
            self.decorators = []


class PythonAnalyzer(ast.NodeVisitor):
    """Analyzes Python files using AST for accurate symbol extraction.

    This analyzer provides the most accurate symbol detection for Python files,
    using Python's built-in AST module to parse the code structure.

    Attributes:
        file_path: Path to the file being analyzed.
        source: Source code content.
        lines: List of source lines.
        symbols: Extracted symbols.
        current_class: Name of class currently being visited (for method detection).
        imports: List of imported modules/names.

    Example:
        >>> source = '''
        ... def greet(name: str) -> str:
        ...     \"\"\"Say hello.\"\"\"
        ...     return f"Hello, {name}"
        ... '''
        >>> analyzer = PythonAnalyzer('example.py', source)
        >>> symbols = analyzer.analyze()
        >>> print(symbols[0].name)
        'greet'
    """

    def __init__(self, file_path: str, source: str):
        """Initialize the Python analyzer.

        Args:
            file_path: Relative path to the file.
            source: Source code content.
        """
        self.file_path = file_path
        self.source = source
        self.lines = source.split("\n")
        self.symbols: list[Symbol] = []
        self.current_class: str | None = None
        self.imports: list[str] = []

    def get_line_end(self, node) -> int:
        """Get the end line of an AST node.

        Args:
            node: An AST node.

        Returns:
            The ending line number of the node.
        """
        if hasattr(node, "end_lineno") and node.end_lineno:
            return node.end_lineno
        if hasattr(node, "body") and node.body:
            last_node = node.body[-1]
            return self.get_line_end(last_node)
        return node.lineno

    def get_signature(self, node) -> str:
        """Extract function/method signature from an AST node.

        Args:
            node: A FunctionDef or AsyncFunctionDef AST node.

        Returns:
            String representation of the function signature.

        Example:
            >>> # For 'async def foo(x: int) -> str:'
            >>> signature = analyzer.get_signature(node)
            >>> print(signature)
            'async def foo(x: int) -> str'
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except (TypeError, AttributeError, RecursionError, ValueError):
                        # ast.unparse can fail on malformed/complex AST nodes
                        pass
                args.append(arg_str)

            returns = ""
            if node.returns:
                try:
                    returns = f" -> {ast.unparse(node.returns)}"
                except (TypeError, AttributeError, RecursionError, ValueError):
                    # ast.unparse can fail on malformed/complex AST nodes
                    pass

            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            return f"{prefix}def {node.name}({', '.join(args)}){returns}"
        return ""

    def get_decorators(self, node) -> list[str]:
        """Extract decorator names from an AST node.

        Args:
            node: An AST node with decorator_list attribute.

        Returns:
            List of decorator name strings.
        """
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except (TypeError, AttributeError, RecursionError, ValueError):
                # Fallback: try to get simple decorator name
                if isinstance(dec, ast.Name):
                    decorators.append(dec.id)
        return decorators

    def get_docstring(self, node) -> str | None:
        """Extract docstring from an AST node, truncated for efficiency.

        Args:
            node: An AST node that may have a docstring.

        Returns:
            First 3 lines of the docstring, or None if no docstring.
        """
        doc = ast.get_docstring(node)
        if doc:
            lines = doc.split("\n")
            if len(lines) > 3:
                return "\n".join(lines[:3]) + "..."
            return doc
        return None

    def visit_Import(self, node):
        """Visit an import statement."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit a from...import statement."""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit a class definition."""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except (TypeError, AttributeError, RecursionError, ValueError):
                # ast.unparse can fail on complex/malformed base class expressions
                pass

        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=self.get_line_end(node),
            signature=signature,
            docstring=self.get_docstring(node),
            decorators=self.get_decorators(node),
        )
        self.symbols.append(symbol)

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit an async function definition."""
        self._visit_function(node)

    def _visit_function(self, node):
        """Process a function or async function definition.

        Args:
            node: A FunctionDef or AsyncFunctionDef AST node.
        """
        symbol_type = "method" if self.current_class else "function"

        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)

        symbol = Symbol(
            name=node.name,
            type=symbol_type,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=self.get_line_end(node),
            signature=self.get_signature(node),
            docstring=self.get_docstring(node),
            parent=self.current_class,
            dependencies=list(set(calls)),
            decorators=self.get_decorators(node),
        )
        self.symbols.append(symbol)
        self.generic_visit(node)

    def analyze(self) -> list[Symbol]:
        """Parse and analyze the file.

        Returns:
            List of Symbol objects found in the file.

        Raises:
            SyntaxError: If the file has invalid Python syntax (caught and logged).
        """
        try:
            tree = ast.parse(self.source)
            self.visit(tree)
        except SyntaxError as e:
            print(f"Syntax error in {self.file_path}: {e}", file=sys.stderr)
        return self.symbols


class GenericAnalyzer:
    """Regex-based analyzer for non-Python languages.

    Provides symbol detection for JavaScript, TypeScript, Java, Go, Rust, and C/C++
    using regular expression patterns. Less accurate than AST analysis but works
    across multiple languages.

    Attributes:
        PATTERNS: Dict of regex patterns for each supported language.
        file_path: Path to the file being analyzed.
        source: Source code content.
        language: The programming language of the file.

    Example:
        >>> source = 'function greet(name) { return "Hello, " + name; }'
        >>> analyzer = GenericAnalyzer('example.js', source, 'javascript')
        >>> symbols = analyzer.analyze()
    """

    PATTERNS = {
        "javascript": {
            "function": r"(?:async\s+)?function\s+(\w+)\s*\([^)]*\)",
            "arrow": r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            "class": r"class\s+(\w+)(?:\s+extends\s+\w+)?",
            "method": r"(?:async\s+)?(\w+)\s*\([^)]*\)\s*{",
        },
        "typescript": {
            "function": r"(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)",
            "interface": r"interface\s+(\w+)",
            "type": r"type\s+(\w+)\s*=",
            "class": r"class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?",
        },
        "java": {
            "class": r"(?:public|private|protected)?\s*class\s+(\w+)",
            "interface": r"interface\s+(\w+)",
            "method": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\([^)]*\)",
        },
        "go": {
            "function": r"func\s+(\w+)\s*(?:\[[^\]]*\])?\s*\(",
            "method": r"func\s+\([^)]+\)\s+(\w+)\s*(?:\[[^\]]*\])?\s*\(",
            "struct": r"type\s+(\w+)\s+struct",
            "interface": r"type\s+(\w+)\s+interface",
            "type_alias": r"type\s+(\w+)\s+(?!struct\b|interface\b)\w+",
        },
        "ruby": {
            "function": r"^[ \t]*def\s+(?!self\.)(\w+[!?=]?)",
            "class": r"^[ \t]*class\s+([A-Z]\w*)",
            "module": r"^[ \t]*module\s+([A-Z]\w*)",
        },
        "rust": {
            "function": r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
            "struct": r"(?:pub\s+)?struct\s+(\w+)",
            "impl": r"impl(?:<[^>]*>)?\s+(\w+)",
            "trait": r"(?:pub\s+)?trait\s+(\w+)",
            "enum": r"(?:pub\s+)?enum\s+(\w+)",
        },
        "dart": {
            "class": r"(?:abstract\s+)?class\s+(\w+)",
            "mixin": r"^[ \t]*mixin\s+(\w+)",
            "enum": r"enum\s+(\w+)\s*\{",
            "extension": r"extension\s+(\w+)\s+on\s+\w+",
            "function": r"^[ \t]*(?:Future<?[^>]*>?|void|String|int|double|bool|dynamic|\w+\??)\s+(\w+)\s*\([^)]*\)\s*(?:async\s*)?\{",
        },
    }

    # Maximum lines to scan for a symbol's end before giving up
    MAX_SYMBOL_LINES = 500

    # Languages that use 'end' keyword instead of braces for block termination
    KEYWORD_END_LANGUAGES = {"ruby"}

    # Keywords that open a new block in end-based languages
    _END_OPENERS = (
        "def ",
        "class ",
        "module ",
        "do",
        "if ",
        "unless ",
        "while ",
        "until ",
        "for ",
        "begin",
        "case ",
    )

    def __init__(self, file_path: str, source: str, language: str):
        """Initialize the generic analyzer.

        Args:
            file_path: Relative path to the file.
            source: Source code content.
            language: Programming language identifier.
        """
        self.file_path = file_path
        self.source = source
        self.language = language
        self.lines = source.split("\n")

    def analyze(self) -> list[Symbol]:
        """Analyze the file using regex patterns.

        Returns:
            List of Symbol objects found in the file.
        """
        import re

        symbols = []
        patterns = self.PATTERNS.get(self.language, {})

        for symbol_type, pattern in patterns.items():
            for match in re.finditer(pattern, self.source, re.MULTILINE):
                name = match.group(1)
                line_num = self.source[: match.start()].count("\n") + 1

                line_end = line_num
                was_truncated = False

                if self.language in self.KEYWORD_END_LANGUAGES:
                    # Keyword-based end detection (Ruby: def/class/module ... end)
                    depth = 1
                    for i, line in enumerate(self.lines[line_num:], start=line_num + 1):
                        stripped = line.strip()
                        if not stripped.startswith("#"):
                            for kw in self._END_OPENERS:
                                if stripped.startswith(kw) or stripped == kw.strip():
                                    depth += 1
                                    break
                            if (
                                stripped == "end"
                                or stripped.startswith("end ")
                                or stripped.startswith("end;")
                            ):
                                depth -= 1
                                if depth <= 0:
                                    line_end = i
                                    break
                        if i > line_num + self.MAX_SYMBOL_LINES:
                            line_end = i
                            was_truncated = True
                            break
                else:
                    # Brace-based end detection (Go, JS, Java, Rust, C/C++)
                    brace_count = 0
                    started = False
                    for i, line in enumerate(self.lines[line_num - 1 :], start=line_num):
                        brace_count += line.count("{") - line.count("}")
                        if "{" in line:
                            started = True
                        if started and brace_count <= 0:
                            line_end = i
                            break
                        if i > line_num + self.MAX_SYMBOL_LINES:
                            line_end = i
                            was_truncated = True
                            break

                symbols.append(
                    Symbol(
                        name=name,
                        type=symbol_type,
                        file_path=self.file_path,
                        line_start=line_num,
                        line_end=line_end,
                        signature=match.group(0).strip()[:100],
                        truncated=was_truncated,
                    )
                )

        return symbols


class GitIntegration:
    """Git integration utilities for the code mapper.

    Provides methods to get git-tracked files, parse .gitignore,
    and find changes since a specific commit.

    Attributes:
        root_path: Path to the git repository root.
        available: Whether git is available and this is a git repo.

    Example:
        >>> git = GitIntegration('/path/to/repo')
        >>> if git.available:
        ...     tracked_files = git.get_tracked_files()
        ...     print(f"Found {len(tracked_files)} tracked files")
    """

    def __init__(self, root_path: Path):
        """Initialize git integration.

        Args:
            root_path: Path to the repository root.
        """
        self.root_path = root_path
        self.available = self._check_git_available()

    def _check_git_available(self) -> bool:
        """Check if git is available and this is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def get_tracked_files(self) -> set[str]:
        """Get all files tracked by git.

        Returns:
            Set of relative file paths tracked by git.
        """
        if not self.available:
            return set()

        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return set()

    def get_gitignore_patterns(self) -> list[str]:
        """Parse .gitignore and return patterns.

        Returns:
            List of gitignore patterns.
        """
        patterns = []
        gitignore_path = self.root_path / ".gitignore"

        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except Exception:
                pass

        return patterns

    def get_files_changed_since(self, commit: str) -> set[str]:
        """Get files that changed since a specific commit.

        Args:
            commit: Git commit hash (7-40 hex characters).

        Returns:
            Set of relative file paths that have changed.

        Raises:
            ValueError: If commit is not a valid hex hash.
        """
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.~^/@{}\-]*", commit):
            raise ValueError(f"Invalid git reference: {commit}")

        if not self.available:
            return set()

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", commit, "HEAD"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return set()

    def get_uncommitted_changes(self) -> set[str]:
        """Get files with uncommitted changes.

        Returns:
            Set of relative file paths with uncommitted changes.
        """
        if not self.available:
            return set()

        try:
            # Get both staged and unstaged changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                files = set()
                for line in result.stdout.strip().split("\n"):
                    if line and len(line) > 3:
                        # Format: "XY filename" where XY is status
                        files.add(line[3:].strip())
                return files
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return set()


class CodeNavigator:
    """Main class for mapping a codebase to create a searchable index.

    Scans a directory tree, analyzes source files, and generates a JSON index
    containing all symbols, their locations, signatures, and dependencies.

    Attributes:
        root_path: Absolute path to the codebase root.
        ignore_patterns: List of patterns to skip during scanning.
        symbols: List of all discovered symbols.
        file_hashes: Dict mapping file paths to content hashes.
        stats: Dict with processing statistics.

    Example:
        >>> mapper = CodeNavigator('/path/to/project')
        >>> code_map = mapper.scan()
        >>> print(f"Found {code_map['stats']['symbols_found']} symbols")
        Found 1847 symbols

        >>> # Save to file
        >>> import json
        >>> with open('.codenav.json', 'w') as f:
        ...     json.dump(code_map, f)
    """

    def __init__(
        self,
        root_path: str,
        ignore_patterns: list[str] = None,
        git_only: bool = False,
        use_gitignore: bool = False,
    ):
        """Initialize the code mapper.

        Args:
            root_path: Path to the root directory to scan.
            ignore_patterns: Additional patterns to ignore. Merged with defaults.
            git_only: If True, only scan files tracked by git.
            use_gitignore: If True, also ignore patterns from .gitignore.
        """
        self.root_path = Path(root_path).resolve()
        self.ignore_patterns = list(ignore_patterns or DEFAULT_IGNORE_PATTERNS)
        self.git_only = git_only
        self.use_gitignore = use_gitignore
        self.symbols: list[Symbol] = []
        self.file_hashes: dict[str, str] = {}
        self.stats = {"files_processed": 0, "symbols_found": 0, "errors": 0}
        self._existing_map: dict[str, Any] | None = None

        # Initialize git integration
        self._git = GitIntegration(self.root_path)
        self._git_tracked_files: set[str] | None = None

        # Add gitignore patterns if requested
        if self.use_gitignore and self._git.available:
            gitignore_patterns = self._git.get_gitignore_patterns()
            self.ignore_patterns.extend(gitignore_patterns)

        # Cache git tracked files if git_only mode
        if self.git_only and self._git.available:
            self._git_tracked_files = self._git.get_tracked_files()

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored during scanning.

        Args:
            path: Path to check.

        Returns:
            True if the path matches any ignore pattern or is not git-tracked.
        """
        path_str = str(path)
        name = path.name

        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            if pattern in path_str:
                return True

        return False

    def _is_git_tracked(self, file_path: Path) -> bool:
        """Check if a file is tracked by git.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file is git-tracked (or git_only mode is disabled).
        """
        if not self.git_only or self._git_tracked_files is None:
            return True

        try:
            rel_path = str(file_path.relative_to(self.root_path))
            return rel_path in self._git_tracked_files
        except ValueError:
            return False

    def get_language(self, file_path: Path) -> str | None:
        """Determine the programming language from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Language identifier string, or None if not recognized.
        """
        ext = file_path.suffix.lower()
        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return lang
        return None

    def hash_file(self, content: str) -> str:
        """Generate a hash for file content.

        Args:
            content: File content string.

        Returns:
            12-character MD5 hash of the content.
        """
        from . import compute_content_hash

        return compute_content_hash(content)

    def analyze_file(self, file_path: Path) -> list[Symbol]:
        """Analyze a single file and extract its symbols.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            List of Symbol objects found in the file.
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            rel_path = str(file_path.relative_to(self.root_path))
            self.file_hashes[rel_path] = self.hash_file(content)

            language = self.get_language(file_path)
            if language == "python":
                analyzer = PythonAnalyzer(rel_path, content)
            elif language == "javascript":
                from .js_ts_analyzer import JavaScriptAnalyzer

                is_jsx = file_path.suffix.lower() in (".jsx",)
                analyzer = JavaScriptAnalyzer(rel_path, content, is_jsx=is_jsx)
            elif language == "typescript":
                from .js_ts_analyzer import TypeScriptAnalyzer

                is_tsx = file_path.suffix.lower() in (".tsx",)
                analyzer = TypeScriptAnalyzer(rel_path, content, is_tsx=is_tsx)
            elif language == "ruby":
                from .ruby_analyzer import RubyAnalyzer

                analyzer = RubyAnalyzer(rel_path, content)
            elif language == "go":
                from .go_analyzer import GoAnalyzer

                analyzer = GoAnalyzer(rel_path, content)
            elif language == "rust":
                from .rust_analyzer import RustAnalyzer

                analyzer = RustAnalyzer(rel_path, content)
            elif language == "dart":
                from .dart_analyzer import DartAnalyzer

                analyzer = DartAnalyzer(rel_path, content)
            elif language:
                analyzer = GenericAnalyzer(rel_path, content, language)
            else:
                return []

            return analyzer.analyze()

        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
            return []

    # Maximum time allowed for a scan operation (seconds)
    SCAN_TIMEOUT = 30

    def scan(self) -> dict[str, Any]:
        """Scan the entire codebase and generate a code map.

        Returns:
            Dict containing the complete code map with files, index, and stats.
            Includes 'scan_timeout': True if the operation was cut short.

        Example:
            >>> mapper = CodeNavigator('/my/project')
            >>> result = mapper.scan()
            >>> print(result.keys())
            dict_keys(['version', 'root', 'generated_at', 'stats', 'files', 'index'])
        """
        mode = "git-tracked files" if self.git_only else "codebase"
        print(f"Scanning {mode} at: {self.root_path}", file=sys.stderr)

        if self.git_only:
            if not self._git.available:
                print("Warning: git not available, scanning all files", file=sys.stderr)
            elif self._git_tracked_files:
                print(f"  Git tracked files: {len(self._git_tracked_files)}", file=sys.stderr)

        scan_start = time.monotonic()
        timed_out = False

        for root, dirs, files in os.walk(self.root_path):
            if time.monotonic() - scan_start > self.SCAN_TIMEOUT:
                timed_out = True
                print("Warning: scan timed out, returning partial results", file=sys.stderr)
                break
            dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file
                if self.should_ignore(file_path):
                    continue

                # Skip if not git-tracked (when git_only mode is enabled)
                if not self._is_git_tracked(file_path):
                    continue

                language = self.get_language(file_path)
                if language:
                    symbols = self.analyze_file(file_path)
                    self.symbols.extend(symbols)
                    self.stats["files_processed"] += 1

        self.stats["symbols_found"] = len(self.symbols)
        if timed_out:
            self.stats["scan_timeout"] = True
        return self.generate_map()

    def get_current_file_hash(self, file_path: Path) -> str | None:
        """Get the hash of a file's current content without full analysis.

        Args:
            file_path: Path to the file.

        Returns:
            Hash string, or None if file cannot be read.
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.hash_file(content)
        except Exception:
            return None

    def scan_incremental(self, existing_map_path: str) -> dict[str, Any]:
        """Incrementally update an existing code map.

        Only re-analyzes files that have changed since the last scan.
        This is much faster than a full scan for large codebases.

        Args:
            existing_map_path: Path to the existing .codenav.json file.

        Returns:
            Dict containing the updated code map.

        Example:
            >>> mapper = CodeNavigator('/my/project')
            >>> result = mapper.scan_incremental('.codenav.json')
            >>> print(result['stats'])
            {'files_processed': 5, 'files_unchanged': 137, 'files_added': 2, ...}
        """
        # Load existing map - only extract 'files' to minimize memory usage
        # The full map can be large; we only need the files dict for comparison
        try:
            with open(existing_map_path, encoding="utf-8") as f:
                existing_map = json.load(f)
                # Extract only what we need, let the rest be garbage collected
                existing_files = existing_map.get("files", {})
                del existing_map  # Explicit cleanup of the full map
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Cannot load existing map ({e}), performing full scan", file=sys.stderr)
            return self.scan()
        print(f"Incremental scan at: {self.root_path}", file=sys.stderr)
        print(f"Existing map has {len(existing_files)} files", file=sys.stderr)

        # Initialize incremental stats
        self.stats = {
            "files_processed": 0,
            "files_unchanged": 0,
            "files_added": 0,
            "files_modified": 0,
            "files_deleted": 0,
            "symbols_found": 0,
            "errors": 0,
        }

        # Track which files we've seen in current scan
        current_files: dict[str, str] = {}  # rel_path -> hash

        # First pass: collect all current files and their hashes
        # Note: Files may be deleted/modified during walk (TOCTOU).
        # We handle this by checking existence and catching exceptions.
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not self.should_ignore(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file
                if self.should_ignore(file_path):
                    continue

                # Skip symlinks to prevent symlink attacks
                try:
                    if file_path.is_symlink():
                        continue
                except OSError:
                    continue

                language = self.get_language(file_path)
                if language:
                    rel_path = str(file_path.relative_to(self.root_path))
                    try:
                        current_hash = self.get_current_file_hash(file_path)
                        if current_hash:
                            current_files[rel_path] = current_hash
                    except OSError:
                        # File disappeared or became inaccessible during scan
                        pass

        # Categorize files
        unchanged_files = []
        modified_files = []
        added_files = []

        for rel_path, current_hash in current_files.items():
            if rel_path in existing_files:
                existing_hash = existing_files[rel_path].get("hash", "")
                if current_hash == existing_hash:
                    unchanged_files.append(rel_path)
                else:
                    modified_files.append(rel_path)
            else:
                added_files.append(rel_path)

        # Files in existing map but not in current scan = deleted
        deleted_files = [f for f in existing_files if f not in current_files]

        print(f"  Unchanged: {len(unchanged_files)}", file=sys.stderr)
        print(f"  Modified: {len(modified_files)}", file=sys.stderr)
        print(f"  Added: {len(added_files)}", file=sys.stderr)
        print(f"  Deleted: {len(deleted_files)}", file=sys.stderr)

        # Preserve unchanged files' symbols
        for rel_path in unchanged_files:
            file_info = existing_files[rel_path]
            self.file_hashes[rel_path] = file_info.get("hash", "")

            # Convert stored symbols back to Symbol objects
            for sym_data in file_info.get("symbols", []):
                symbol = Symbol(
                    name=sym_data["name"],
                    type=sym_data["type"],
                    file_path=rel_path,
                    line_start=sym_data["lines"][0],
                    line_end=sym_data["lines"][1],
                    signature=sym_data.get("signature"),
                    docstring=sym_data.get("docstring"),
                    parent=sym_data.get("parent"),
                    dependencies=sym_data.get("deps") or [],
                    decorators=sym_data.get("decorators") or [],
                    truncated=sym_data.get("truncated", False),
                )
                self.symbols.append(symbol)

        self.stats["files_unchanged"] = len(unchanged_files)

        # Analyze modified and added files
        # Note: TOCTOU mitigation - files may have changed or been deleted
        # between the hash check and analysis. We handle this gracefully.
        files_to_analyze = modified_files + added_files
        for rel_path in files_to_analyze:
            file_path = self.root_path / rel_path
            try:
                # Check file still exists and is a regular file (not symlink)
                if not file_path.is_file() or file_path.is_symlink():
                    # File was deleted or replaced with symlink between hash and analyze
                    print(
                        f"  Skipping {rel_path}: file no longer exists or is symlink",
                        file=sys.stderr,
                    )
                    self.stats["errors"] += 1
                    continue

                symbols = self.analyze_file(file_path)
                self.symbols.extend(symbols)
                self.stats["files_processed"] += 1
            except OSError as e:
                # File became inaccessible between hash check and analysis (TOCTOU)
                print(f"  Skipping {rel_path}: {e}", file=sys.stderr)
                self.stats["errors"] += 1
                continue

        self.stats["files_added"] = len(added_files)
        self.stats["files_modified"] = len(modified_files)
        self.stats["files_deleted"] = len(deleted_files)
        self.stats["symbols_found"] = len(self.symbols)

        return self.generate_map()

    def generate_map(self) -> dict[str, Any]:
        """Generate the code map structure from collected symbols.

        Returns:
            Dict with version, root, timestamp, stats, files map, and symbol index.
        """
        # Start with all analyzed files (including those with no symbols)
        files_map = {}
        for file_path, file_hash in self.file_hashes.items():
            files_map[file_path] = {
                "hash": file_hash,
                "symbols": [],
            }

        # Add symbols to their respective files
        for symbol in self.symbols:
            if symbol.file_path not in files_map:
                files_map[symbol.file_path] = {
                    "hash": self.file_hashes.get(symbol.file_path, ""),
                    "symbols": [],
                }
            symbol_dict = {
                "name": symbol.name,
                "type": symbol.type,
                "lines": [symbol.line_start, symbol.line_end],
                "signature": symbol.signature,
                "docstring": symbol.docstring,
                "parent": symbol.parent,
                "deps": symbol.dependencies[:10] if symbol.dependencies else None,
                "decorators": symbol.decorators if symbol.decorators else None,
            }
            # Only include truncated flag when True (keeps output compact)
            if symbol.truncated:
                symbol_dict["truncated"] = True
            files_map[symbol.file_path]["symbols"].append(symbol_dict)

        symbol_index = {}
        for symbol in self.symbols:
            key = symbol.name.lower()
            if key not in symbol_index:
                symbol_index[key] = []
            symbol_index[key].append(
                {
                    "file": symbol.file_path,
                    "type": symbol.type,
                    "lines": [symbol.line_start, symbol.line_end],
                    "parent": symbol.parent,
                }
            )

        return {
            "version": "1.0",
            "root": str(self.root_path),
            "generated_at": datetime.now().isoformat(),
            "stats": self.stats,
            "files": files_map,
            "index": symbol_index,
        }


def add_map_arguments(parser: argparse.ArgumentParser) -> None:
    """Add map command arguments to a parser.

    Args:
        parser: The argument parser to add arguments to.
    """
    parser.add_argument("path", help="Path to the codebase root directory")
    parser.add_argument(
        "-o", "--output", default=".codenav.json", help="Output file path (default: .codenav.json)"
    )
    parser.add_argument("-i", "--ignore", nargs="*", help="Additional patterns to ignore")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only update changed files (requires existing map)",
    )
    parser.add_argument(
        "--git-only",
        action="store_true",
        help="Only scan files tracked by git",
    )
    parser.add_argument(
        "--use-gitignore",
        action="store_true",
        help="Also ignore patterns from .gitignore",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Output compact JSON (default: pretty-printed)"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")


def run_map(args: argparse.Namespace) -> None:
    """Execute the map command with parsed arguments.

    Args:
        args: Parsed command-line arguments.
    """
    ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
    if args.ignore:
        ignore_patterns.extend(args.ignore)

    git_only = getattr(args, "git_only", False)
    use_gitignore = getattr(args, "use_gitignore", False)

    mapper = CodeNavigator(
        args.path,
        ignore_patterns,
        git_only=git_only,
        use_gitignore=use_gitignore,
    )

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.path, output_path)

    # Use incremental scan if requested and existing map exists
    incremental = getattr(args, "incremental", False)
    if incremental and os.path.exists(output_path):
        code_map = mapper.scan_incremental(output_path)
    else:
        if incremental:
            print(f"No existing map at {output_path}, performing full scan", file=sys.stderr)
        code_map = mapper.scan()

    with open(output_path, "w", encoding="utf-8") as f:
        if args.compact:
            json.dump(code_map, f, separators=(",", ":"))
        else:
            json.dump(code_map, f, indent=2)

    c = get_colors(no_color=args.no_color)
    stats = code_map["stats"]

    # Display appropriate message based on scan type
    if "files_unchanged" in stats:
        # Incremental scan
        print(f"\n{c.success('✓')} Code map updated: {c.cyan(output_path)}", file=sys.stderr)
        print(f"  Unchanged: {c.dim(str(stats['files_unchanged']))}", file=sys.stderr)
        print(f"  Modified: {c.yellow(str(stats['files_modified']))}", file=sys.stderr)
        print(f"  Added: {c.green(str(stats['files_added']))}", file=sys.stderr)
        print(f"  Deleted: {c.magenta(str(stats['files_deleted']))}", file=sys.stderr)
        print(f"  Total symbols: {c.green(str(stats['symbols_found']))}", file=sys.stderr)
    else:
        # Full scan
        print(f"\n{c.success('✓')} Code map generated: {c.cyan(output_path)}", file=sys.stderr)
        print(f"  Files processed: {c.green(str(stats['files_processed']))}", file=sys.stderr)
        print(f"  Symbols found: {c.green(str(stats['symbols_found']))}", file=sys.stderr)

    summary = {"output": output_path, "stats": stats}
    if args.compact:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        print(json.dumps(summary, indent=2))


def main():
    """Command-line interface for the code mapper.

    Usage:
        codenav scan /path/to/project [-o OUTPUT] [-i IGNORE...] [--compact]

    Example:
        $ codenav scan /my/project -o .codenav.json
    """
    parser = argparse.ArgumentParser(
        description="Generate a code map for token-efficient navigation",
        epilog="Example: codenav scan /my/project -o .codenav.json",
    )
    add_map_arguments(parser)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()
    run_map(args)


if __name__ == "__main__":
    main()
