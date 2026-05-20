"""Tests for DartAnalyzer — covers both tree-sitter and regex fallback paths."""

import pytest

from codenav.dart_analyzer import TREE_SITTER_AVAILABLE, DartAnalyzer
from codenav.code_navigator import GenericAnalyzer


SAMPLE = open("tests/fixtures/sample_dart.dart").read()


def _names(symbols, type_=None):
    if type_:
        return [s.name for s in symbols if s.type == type_]
    return [s.name for s in symbols]


class TestDartAnalyzerFallback:
    """GenericAnalyzer regex fallback — always runs regardless of tree-sitter."""

    def test_detects_classes(self):
        analyzer = GenericAnalyzer("sample.dart", SAMPLE, "dart")
        symbols = analyzer.analyze()
        classes = _names(symbols, "class")
        assert "MyWidget" in classes
        assert "Counter" in classes

    def test_detects_enum(self):
        analyzer = GenericAnalyzer("sample.dart", SAMPLE, "dart")
        symbols = analyzer.analyze()
        assert "Status" in _names(symbols, "enum")

    def test_detects_mixin(self):
        analyzer = GenericAnalyzer("sample.dart", SAMPLE, "dart")
        symbols = analyzer.analyze()
        assert "Loggable" in _names(symbols, "mixin")

    def test_detects_extension(self):
        analyzer = GenericAnalyzer("sample.dart", SAMPLE, "dart")
        symbols = analyzer.analyze()
        assert "StringExtension" in _names(symbols, "extension")

    def test_line_numbers_positive(self):
        analyzer = GenericAnalyzer("sample.dart", SAMPLE, "dart")
        for s in analyzer.analyze():
            assert s.line_start >= 1
            assert s.line_end >= s.line_start


class TestDartAnalyzer:
    """DartAnalyzer — uses tree-sitter when available, regex otherwise."""

    def test_analyze_returns_symbols(self):
        analyzer = DartAnalyzer("sample.dart", SAMPLE)
        symbols = analyzer.analyze()
        assert len(symbols) > 0

    def test_detects_classes(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        classes = _names(symbols, "class")
        assert "MyWidget" in classes
        assert "Counter" in classes
        assert "_CounterState" in classes

    def test_detects_enum(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        assert "Status" in _names(symbols, "enum")

    def test_detects_mixin(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        assert "Loggable" in _names(symbols, "mixin")

    def test_detects_extension(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        assert "StringExtension" in _names(symbols, "extension")

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter-dart not installed")
    def test_methods_have_parent(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        methods = [s for s in symbols if s.type == "method"]
        assert any(s.parent == "MyWidget" and s.name == "build" for s in methods)
        assert any(s.parent == "_CounterState" and s.name == "_increment" for s in methods)

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter-dart not installed")
    def test_top_level_function(self):
        symbols = DartAnalyzer("sample.dart", SAMPLE).analyze()
        assert "formatCurrency" in _names(symbols, "function")

    def test_empty_file(self):
        symbols = DartAnalyzer("empty.dart", "").analyze()
        assert symbols == []

    def test_invalid_dart_does_not_crash(self):
        bad = "class { broken dart *** @@@ }"
        symbols = DartAnalyzer("bad.dart", bad).analyze()
        assert isinstance(symbols, list)

    def test_line_numbers_positive(self):
        for s in DartAnalyzer("sample.dart", SAMPLE).analyze():
            assert s.line_start >= 1
            assert s.line_end >= s.line_start
