"""Tests for Text Preprocessor & Sanitizer."""

from app.analysis.preprocessor import preprocess
from app.analysis.types import SanitizationFlag


class TestEmptyInput:
    """Test empty / malformed input handling."""

    def test_empty_string(self):
        result = preprocess("")
        assert result.flag == SanitizationFlag.EMPTY_RESPONSE
        assert result.text == ""

    def test_none_like_whitespace(self):
        result = preprocess("   \n\n  \t  ")
        assert result.flag == SanitizationFlag.EMPTY_RESPONSE

    def test_normal_text_passes_through(self):
        result = preprocess("Hello world, this is a normal response.")
        assert result.flag == SanitizationFlag.CLEAN
        assert "Hello world" in result.text


class TestDeepSeekThink:
    """Test DeepSeek <think> block stripping."""

    def test_simple_think_block(self):
        text = "<think>Let me think about this...</think>The answer is 42."
        result = preprocess(text, vendor="deepseek")
        assert result.flag == SanitizationFlag.DEEPSEEK_THINK_STRIPPED
        assert "<think>" not in result.text
        assert "The answer is 42" in result.text
        assert "Let me think" in result.think_content

    def test_multiline_think_block(self):
        text = """<think>
Step 1: Consider the brands.
Step 2: Evaluate prices.
Step 3: Formulate response.
</think>

Based on my analysis, here are the top banks."""
        result = preprocess(text, vendor="deepseek")
        assert result.flag == SanitizationFlag.DEEPSEEK_THINK_STRIPPED
        assert "top banks" in result.text
        assert "Step 1" in result.think_content

    def test_think_block_only_is_empty(self):
        text = "<think>Internal reasoning only, no actual response.</think>"
        result = preprocess(text, vendor="deepseek")
        assert result.flag == SanitizationFlag.EMPTY_RESPONSE

    def test_think_auto_detected_without_vendor(self):
        text = "<think>reasoning</think>Actual content here."
        result = preprocess(text, vendor="")
        assert result.flag == SanitizationFlag.DEEPSEEK_THINK_STRIPPED
        assert "Actual content" in result.text

    def test_multiple_think_blocks(self):
        text = "<think>First thought</think>Content<think>Second thought</think>More content"
        result = preprocess(text)
        assert "First thought" in result.think_content
        assert "Second thought" in result.think_content
        assert "Content" in result.text
        assert "More content" in result.text


class TestCensoredMarkers:
    """Test vendor censorship detection."""

    def test_censored_by_vendor(self):
        result = preprocess("[CENSORED_BY_VENDOR]")
        assert result.flag == SanitizationFlag.CENSORED

    def test_censored_marker(self):
        result = preprocess("Some text [CENSORED] more text")
        assert result.flag == SanitizationFlag.CENSORED

    def test_blocked_marker(self):
        result = preprocess("[BLOCKED]")
        assert result.flag == SanitizationFlag.CENSORED


class TestRefusalDetection:
    """Test vendor guardrail / refusal detection."""

    def test_russian_ai_disclaimer(self):
        text = "Я ИИ и не могу давать финансовых рекомендаций."
        result = preprocess(text)
        assert result.flag == SanitizationFlag.VENDOR_REFUSAL

    def test_russian_consult_specialist(self):
        text = "Обратитесь к специалисту для получения точной информации. Я языковая модель."
        result = preprocess(text)
        assert result.flag == SanitizationFlag.VENDOR_REFUSAL

    def test_english_ai_disclaimer(self):
        text = "I'm an AI and cannot provide financial advice."
        result = preprocess(text)
        assert result.flag == SanitizationFlag.VENDOR_REFUSAL

    def test_english_consult_professional(self):
        text = "I am unable to provide medical advice. Please consult a doctor."
        result = preprocess(text)
        assert result.flag == SanitizationFlag.VENDOR_REFUSAL

    def test_single_refusal_in_long_text_passes(self):
        # One refusal pattern in a long informative text should pass through
        text = (
            "Я языковая модель. Вот топ-5 банков для бизнеса:\n"
            "1. Тинькофф — быстрое открытие счёта\n"
            "2. Сбербанк — широкая сеть отделений\n"
            "3. Альфа-Банк — удобное мобильное приложение\n"
            "4. Модульбанк — лучший для ИП\n"
            "5. Точка — отличная техподдержка"
        )
        result = preprocess(text)
        # Long text with useful content shouldn't be flagged as refusal
        assert result.flag == SanitizationFlag.CLEAN


class TestMarkdownCleanup:
    """Test Markdown artifact removal."""

    def test_bold_markers_removed(self):
        text = "**Тинькофф** — это лучший **банк** для бизнеса."
        result = preprocess(text)
        assert "**" not in result.text
        assert "Тинькофф" in result.text

    def test_italic_markers_removed(self):
        text = "Банк *Тинькофф* предлагает *лучшие* условия."
        result = preprocess(text)
        assert "*" not in result.text
        assert "Тинькофф" in result.text

    def test_excessive_blank_lines_collapsed(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = preprocess(text)
        assert "\n\n\n" not in result.text
        assert "Line 1" in result.text
        assert "Line 2" in result.text

    def test_stripped_chars_counted(self):
        text = "**Bold** text with *italic* markers"
        result = preprocess(text)
        assert result.stripped_chars > 0

    def test_original_text_preserved(self):
        text = "**Original** with *markdown*"
        result = preprocess(text)
        assert result.original_text == text
        assert "**" not in result.text
