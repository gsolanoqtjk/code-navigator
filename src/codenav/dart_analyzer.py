"""Dart/Flutter analyzer using tree-sitter for AST-based symbol extraction.

Falls back to regex-based GenericAnalyzer when tree-sitter-dart is not installed.

Example:
    >>> from codenav.dart_analyzer import DartAnalyzer
    >>> source = '''
    ... class MyWidget extends StatelessWidget {
    ...   Widget build(BuildContext context) {
    ...     return Container();
    ...   }
    ... }
    ... '''
    >>> analyzer = DartAnalyzer('my_widget.dart', source)
    >>> symbols = analyzer.analyze()
    >>> print(symbols[0].name)
    'MyWidget'

Installation:
    To enable AST support, install with the 'ast' extra:
        pip install code-navigator[ast]
"""

import ctypes
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node

try:
    import warnings
    from tree_sitter import Language, Parser

    _dylib = Path(__file__).parent / "dart.dylib"
    _lib = ctypes.CDLL(str(_dylib))
    _lib.tree_sitter_dart.restype = ctypes.c_void_p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _DART_LANGUAGE = Language(_lib.tree_sitter_dart())

    TREE_SITTER_AVAILABLE = True
except Exception:
    TREE_SITTER_AVAILABLE = False

from .code_navigator import GenericAnalyzer, Symbol


class DartAnalyzer:
    """Analyzes Dart/Flutter files using tree-sitter for accurate symbol extraction.

    When tree-sitter-dart is not available, automatically falls back to
    regex-based GenericAnalyzer.

    Attributes:
        file_path: Path to the file being analyzed.
        source: Source code content.
        symbols: Extracted symbols.
    """

    def __init__(self, file_path: str, source: str):
        self.file_path = file_path
        self.source = source
        self.lines = source.split("\n")
        self.symbols: list[Symbol] = []
        self._current_class: str | None = None

    def analyze(self) -> list[Symbol]:
        """Parse and analyze the file.

        Returns:
            List of Symbol objects found in the file.
        """
        if not TREE_SITTER_AVAILABLE:
            return GenericAnalyzer(self.file_path, self.source, "dart").analyze()

        try:
            parser = Parser(_DART_LANGUAGE)
            tree = parser.parse(bytes(self.source, "utf-8"))
            self._visit_node(tree.root_node)
        except Exception as e:
            print(f"tree-sitter error in {self.file_path}: {e}", file=sys.stderr)
            return GenericAnalyzer(self.file_path, self.source, "dart").analyze()

        return self.symbols

    def _visit_node(self, node: "Node") -> None:
        """Recursively visit AST nodes and extract symbols."""
        nt = node.type

        if nt == "class_definition":
            self._extract_class(node)
            return  # Body visited inside _extract_class
        elif nt == "mixin_declaration":
            self._extract_mixin(node)
            return  # Body visited inside _extract_mixin
        elif nt == "enum_declaration":
            self._extract_enum(node)
        elif nt == "extension_declaration":
            self._extract_extension(node)
            return  # Body visited inside _extract_extension
        elif nt == "function_signature":
            # Top-level function (not inside a class/mixin/extension)
            if self._current_class is None:
                self._extract_function(node)
        elif nt == "method_signature":
            # Method inside class/mixin/extension body
            self._extract_method(node)
        elif nt in ("constant_constructor_signature", "constructor_signature"):
            self._extract_constructor(node)
        else:
            for child in node.children:
                self._visit_node(child)
            return

        for child in node.children:
            self._visit_node(child)

    def _get_text(self, node: "Node") -> str:
        return self.source[node.start_byte : node.end_byte]

    def _child_by_type(self, node: "Node", *types: str) -> "Node | None":
        for child in node.children:
            if child.type in types:
                return child
        return None

    def _extract_class(self, node: "Node") -> None:
        name_node = self._child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_text(name_node)
        is_abstract = any(c.type == "abstract" for c in node.children)
        prefix = "abstract " if is_abstract else ""

        sig = f"{prefix}class {name}"
        supertype = self._child_by_type(node, "supertype")
        if supertype:
            sig += f" {self._get_text(supertype)}"

        self.symbols.append(
            Symbol(
                name=name,
                type="class",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig[:100],
            )
        )

        old = self._current_class
        self._current_class = name
        body = self._child_by_type(node, "class_body")
        if body:
            for child in body.children:
                self._visit_node(child)
        self._current_class = old

    def _extract_mixin(self, node: "Node") -> None:
        name_node = self._child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_text(name_node)
        self.symbols.append(
            Symbol(
                name=name,
                type="mixin",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"mixin {name}",
            )
        )

        old = self._current_class
        self._current_class = name
        body = self._child_by_type(node, "class_body")
        if body:
            for child in body.children:
                self._visit_node(child)
        self._current_class = old

    def _extract_enum(self, node: "Node") -> None:
        name_node = self._child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_text(name_node)
        self.symbols.append(
            Symbol(
                name=name,
                type="enum",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"enum {name}",
            )
        )

    def _extract_extension(self, node: "Node") -> None:
        name_node = self._child_by_type(node, "identifier")
        name = self._get_text(name_node) if name_node else "<anonymous>"

        on_type = ""
        for i, child in enumerate(node.children):
            if child.type == "on":
                rest = node.children[i + 1 :]
                if rest:
                    on_type = f" on {self._get_text(rest[0])}"
                break

        self.symbols.append(
            Symbol(
                name=name,
                type="extension",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"extension {name}{on_type}"[:100],
            )
        )

        old = self._current_class
        self._current_class = name
        body = self._child_by_type(node, "class_body")
        if body:
            for child in body.children:
                self._visit_node(child)
        self._current_class = old

    def _extract_function(self, node: "Node") -> None:
        # node is function_signature: [ret_type] identifier formal_parameter_list
        name_node = self._child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._get_text(name_node)
        ret_type = ""
        for child in node.children:
            if child.type in ("type_identifier", "void_type", "nullable_type"):
                ret_type = self._get_text(child) + " "
                break

        self.symbols.append(
            Symbol(
                name=name,
                type="function",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"{ret_type}{name}()"[:100],
            )
        )

    def _extract_method(self, node: "Node") -> None:
        # node is method_signature which wraps function_signature
        func_sig = self._child_by_type(node, "function_signature")
        target = func_sig if func_sig else node
        name_node = self._child_by_type(target, "identifier")
        if not name_node:
            return

        name = self._get_text(name_node)
        ret_type = ""
        for child in target.children:
            if child.type in ("type_identifier", "void_type", "nullable_type"):
                ret_type = self._get_text(child) + " "
                break

        self.symbols.append(
            Symbol(
                name=name,
                type="method",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"{ret_type}{name}()"[:100],
                parent=self._current_class,
            )
        )

    def _extract_constructor(self, node: "Node") -> None:
        # constant_constructor_signature: const? TypeName [.namedConstructor] params
        # First identifier is the class name (e.g. MyWidget), skip it if matches parent
        identifiers = [c for c in node.children if c.type == "identifier"]
        if not identifiers:
            return

        # Use the first identifier as the constructor name
        name = self._get_text(identifiers[0])
        self.symbols.append(
            Symbol(
                name=name,
                type="constructor",
                file_path=self.file_path,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"{name}()",
                parent=self._current_class,
            )
        )
