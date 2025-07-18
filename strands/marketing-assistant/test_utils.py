"""
Unit tests for the utils module.

This module contains comprehensive tests for all utility functions
to ensure they work correctly and handle edge cases properly.
"""

import unittest
from unittest.mock import patch
import json
from datetime import datetime
from utils import (
    clean_and_validate_post,
    optimize_post_length,
    parse_bedrock_response,
    validate_analysis_input,
    validate_pdf_input,
    create_error_response,
    create_success_response,
    extract_json_from_text,
    sanitize_text_content,
    get_current_timestamp
)


class TestCleanAndValidatePost(unittest.TestCase):
    """Test cases for clean_and_validate_post function."""
    
    def test_basic_cleaning(self):
        """Test basic whitespace and newline cleaning."""
        input_text = "  This is a test   \n\n  post  "
        expected = "This is a test post"
        result = clean_and_validate_post(input_text)
        self.assertEqual(result, expected)
    
    def test_quote_removal(self):
        """Test removal of wrapping quotes."""
        # Double quotes
        input_text = '"This is a quoted post"'
        expected = "This is a quoted post"
        result = clean_and_validate_post(input_text)
        self.assertEqual(result, expected)
        
        # Single quotes
        input_text = "'This is a quoted post'"
        expected = "This is a quoted post"
        result = clean_and_validate_post(input_text)
        self.assertEqual(result, expected)
    
    def test_prefix_removal(self):
        """Test removal of LLM-generated prefixes."""
        test_cases = [
            ("Here's a social media post: Great content!", "Great content!"),
            ("Social media post: Amazing insights", "Amazing insights"),
            ("Tweet: Short and sweet", "Short and sweet"),
            ("Post: Regular content", "Regular content"),
            ("Here's the post: Final version", "Final version"),
            ("Generated post: Auto content", "Auto content")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = clean_and_validate_post(input_text)
                self.assertEqual(result, expected)
    
    def test_case_insensitive_prefix_removal(self):
        """Test that prefix removal is case insensitive."""
        input_text = "TWEET: This should work"
        expected = "This should work"
        result = clean_and_validate_post(input_text)
        self.assertEqual(result, expected)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = clean_and_validate_post("")
        self.assertEqual(result, "")
        
        result = clean_and_validate_post("   ")
        self.assertEqual(result, "")


class TestOptimizePostLength(unittest.TestCase):
    """Test cases for optimize_post_length function."""
    
    def test_short_post_unchanged(self):
        """Test that short posts are not modified."""
        short_post = "This is a short post."
        themes = ["test"]
        result_post, notes = optimize_post_length(short_post, themes)
        self.assertEqual(result_post, short_post)
        self.assertEqual(notes, [])
    
    def test_aggressive_truncation(self):
        """Test aggressive truncation for very long posts."""
        very_long_post = "a" * 400  # 400 characters
        themes = ["test"]
        result_post, notes = optimize_post_length(very_long_post, themes)
        self.assertLessEqual(len(result_post), 280)
        self.assertIn("Aggressive truncation applied", notes)
    
    def test_punctuation_optimization(self):
        """Test punctuation optimization."""
        post_with_punctuation = "This is great!!! Really amazing... What do you think???"
        themes = ["test"]
        result_post, notes = optimize_post_length(post_with_punctuation, themes)
        
        # Should reduce multiple punctuation marks
        self.assertNotIn("!!!", result_post)
        self.assertNotIn("???", result_post)
        self.assertIn("!", result_post)
        self.assertIn("?", result_post)
    
    def test_contraction_optimization(self):
        """Test contraction of common phrases."""
        post_with_contractions = "You are right that is good and it is working"
        themes = ["test"]
        result_post, notes = optimize_post_length(post_with_contractions, themes)
        
        # Should contract phrases
        self.assertIn("you're", result_post.lower())
        self.assertIn("that's", result_post.lower())
        self.assertIn("it's", result_post.lower())
    
    def test_filler_word_removal(self):
        """Test removal of filler words."""
        post_with_fillers = "This is very really quite actually good content"
        themes = ["test"]
        result_post, notes = optimize_post_length(post_with_fillers, themes)
        
        # Should remove filler words
        self.assertNotIn("very", result_post.lower())
        self.assertNotIn("really", result_post.lower())
        self.assertNotIn("quite", result_post.lower())
        self.assertNotIn("actually", result_post.lower())
    
    def test_word_boundary_truncation(self):
        """Test truncation at word boundaries."""
        # Create a post that's just over 280 characters
        long_post = "This is a test post that needs to be truncated because it is too long for social media platforms and exceeds the character limit that we have set for posts which should be under two hundred and eighty characters total"
        themes = ["test"]
        result_post, notes = optimize_post_length(long_post, themes)
        
        self.assertLessEqual(len(result_post), 280)
        # Should end with complete word or ellipsis
        self.assertTrue(result_post.endswith("...") or result_post.split()[-1].isalpha())


class TestParseBedRockResponse(unittest.TestCase):
    """Test cases for parse_bedrock_response function."""
    
    def test_success_prefix_parsing(self):
        """Test parsing response with 'Success: ' prefix."""
        mock_response = "Success: {'body': {'content': [{'text': 'Parsed content'}]}}"
        result = parse_bedrock_response(mock_response)
        self.assertEqual(result, "Parsed content")
    
    def test_direct_response_parsing(self):
        """Test parsing direct response without prefix."""
        direct_response = "Direct response content"
        result = parse_bedrock_response(direct_response)
        self.assertEqual(result, "Direct response content")
    
    def test_fallback_parsing(self):
        """Test fallback parsing when structured parsing fails."""
        malformed_response = "Success: {malformed json"
        result = parse_bedrock_response(malformed_response)
        # Should fallback to original text
        self.assertEqual(result, malformed_response)
    
    def test_whitespace_stripping(self):
        """Test that whitespace is properly stripped."""
        response_with_whitespace = "  Content with whitespace  "
        result = parse_bedrock_response(response_with_whitespace)
        self.assertEqual(result, "Content with whitespace")


class TestValidateAnalysisInput(unittest.TestCase):
    """Test cases for validate_analysis_input function."""
    
    def test_valid_analysis_result(self):
        """Test validation of valid analysis result."""
        valid_result = {
            'key_message': 'This is a key message',
            'context_summary': 'This is context',
            'core_themes': ['theme1', 'theme2']
        }
        self.assertTrue(validate_analysis_input(valid_result))
    
    def test_valid_with_only_key_message(self):
        """Test validation with only key message."""
        result_with_key_message = {
            'key_message': 'This is a key message',
            'context_summary': '',
            'core_themes': []
        }
        self.assertTrue(validate_analysis_input(result_with_key_message))
    
    def test_valid_with_only_context_summary(self):
        """Test validation with only context summary."""
        result_with_context = {
            'key_message': '',
            'context_summary': 'This is context',
            'core_themes': []
        }
        self.assertTrue(validate_analysis_input(result_with_context))
    
    def test_invalid_empty_dict(self):
        """Test validation of empty dictionary."""
        self.assertFalse(validate_analysis_input({}))
    
    def test_invalid_none_input(self):
        """Test validation of None input."""
        self.assertFalse(validate_analysis_input(None))
    
    def test_invalid_non_dict_input(self):
        """Test validation of non-dictionary input."""
        self.assertFalse(validate_analysis_input("not a dict"))
        self.assertFalse(validate_analysis_input(123))
        self.assertFalse(validate_analysis_input([]))
    
    def test_invalid_empty_content(self):
        """Test validation when both key fields are empty."""
        empty_content_result = {
            'key_message': '',
            'context_summary': '   ',  # Only whitespace
            'core_themes': ['theme1']
        }
        self.assertFalse(validate_analysis_input(empty_content_result))


class TestValidatePdfInput(unittest.TestCase):
    """Test cases for validate_pdf_input function."""
    
    def test_valid_file_path_only(self):
        """Test validation with valid file path only."""
        is_valid, error = validate_pdf_input("test.pdf")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_valid_file_path_and_start_page(self):
        """Test validation with valid file path and start page."""
        is_valid, error = validate_pdf_input("test.pdf", 5)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_invalid_empty_file_path(self):
        """Test validation with empty file path."""
        is_valid, error = validate_pdf_input("")
        self.assertFalse(is_valid)
        self.assertIn("Invalid file_path", error)
    
    def test_invalid_none_file_path(self):
        """Test validation with None file path."""
        is_valid, error = validate_pdf_input(None)
        self.assertFalse(is_valid)
        self.assertIn("Invalid file_path", error)
    
    def test_invalid_non_string_file_path(self):
        """Test validation with non-string file path."""
        is_valid, error = validate_pdf_input(123)
        self.assertFalse(is_valid)
        self.assertIn("Invalid file_path", error)
    
    def test_invalid_zero_start_page(self):
        """Test validation with zero start page."""
        is_valid, error = validate_pdf_input("test.pdf", 0)
        self.assertFalse(is_valid)
        self.assertIn("Invalid start_page", error)
    
    def test_invalid_negative_start_page(self):
        """Test validation with negative start page."""
        is_valid, error = validate_pdf_input("test.pdf", -1)
        self.assertFalse(is_valid)
        self.assertIn("Invalid start_page", error)
    
    def test_invalid_non_integer_start_page(self):
        """Test validation with non-integer start page."""
        is_valid, error = validate_pdf_input("test.pdf", "5")
        self.assertFalse(is_valid)
        self.assertIn("Invalid start_page", error)


class TestCreateErrorResponse(unittest.TestCase):
    """Test cases for create_error_response function."""
    
    def test_basic_error_response(self):
        """Test creation of basic error response."""
        error_msg = "Test error message"
        result = create_error_response(error_msg)
        
        expected = {
            'success': False,
            'error': error_msg
        }
        self.assertEqual(result, expected)
    
    def test_error_response_with_additional_fields(self):
        """Test creation of error response with additional fields."""
        error_msg = "Test error"
        result = create_error_response(error_msg, code=404, details="Not found")
        
        expected = {
            'success': False,
            'error': error_msg,
            'code': 404,
            'details': "Not found"
        }
        self.assertEqual(result, expected)


class TestCreateSuccessResponse(unittest.TestCase):
    """Test cases for create_success_response function."""
    
    def test_basic_success_response(self):
        """Test creation of basic success response."""
        result = create_success_response()
        
        expected = {
            'success': True,
            'error': None
        }
        self.assertEqual(result, expected)
    
    def test_success_response_with_fields(self):
        """Test creation of success response with additional fields."""
        result = create_success_response(data="test data", count=5)
        
        expected = {
            'success': True,
            'error': None,
            'data': "test data",
            'count': 5
        }
        self.assertEqual(result, expected)


class TestExtractJsonFromText(unittest.TestCase):
    """Test cases for extract_json_from_text function."""
    
    def test_extract_valid_json(self):
        """Test extraction of valid JSON from text."""
        text_with_json = 'Here is some JSON: {"key": "value", "number": 42}'
        result = extract_json_from_text(text_with_json)
        
        expected = {"key": "value", "number": 42}
        self.assertEqual(result, expected)
    
    def test_extract_complex_json(self):
        """Test extraction of complex JSON from text."""
        text_with_json = 'Response: {"themes": ["theme1", "theme2"], "score": 0.85}'
        result = extract_json_from_text(text_with_json)
        
        expected = {"themes": ["theme1", "theme2"], "score": 0.85}
        self.assertEqual(result, expected)
    
    def test_no_json_in_text(self):
        """Test handling of text without JSON."""
        text_without_json = "This is just regular text without JSON"
        result = extract_json_from_text(text_without_json)
        self.assertIsNone(result)
    
    def test_invalid_json_in_text(self):
        """Test handling of invalid JSON in text."""
        text_with_invalid_json = "Here is invalid JSON: {key: value, invalid}"
        result = extract_json_from_text(text_with_invalid_json)
        self.assertIsNone(result)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = extract_json_from_text("")
        self.assertIsNone(result)


class TestSanitizeTextContent(unittest.TestCase):
    """Test cases for sanitize_text_content function."""
    
    def test_basic_sanitization(self):
        """Test basic whitespace sanitization."""
        messy_text = "  This   has    excessive   whitespace  \n\n  "
        result = sanitize_text_content(messy_text)
        expected = "This has excessive whitespace"
        self.assertEqual(result, expected)
    
    def test_truncation_with_max_length(self):
        """Test truncation when max_length is specified."""
        long_text = "This is a very long text that should be truncated"
        result = sanitize_text_content(long_text, max_length=20)
        
        self.assertLessEqual(len(result), 20)
        self.assertTrue(result.endswith("..."))
    
    def test_no_truncation_when_under_limit(self):
        """Test no truncation when text is under max_length."""
        short_text = "Short text"
        result = sanitize_text_content(short_text, max_length=50)
        self.assertEqual(result, short_text)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = sanitize_text_content("")
        self.assertEqual(result, "")
        
        result = sanitize_text_content(None)
        self.assertEqual(result, "")


class TestGetCurrentTimestamp(unittest.TestCase):
    """Test cases for get_current_timestamp function."""
    
    def test_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        timestamp = get_current_timestamp()
        
        # Should be able to parse as ISO format
        try:
            parsed_time = datetime.fromisoformat(timestamp)
            self.assertIsInstance(parsed_time, datetime)
        except ValueError:
            self.fail("Timestamp is not in valid ISO format")
    
    def test_timestamp_is_recent(self):
        """Test that timestamp is recent (within last few seconds)."""
        timestamp = get_current_timestamp()
        parsed_time = datetime.fromisoformat(timestamp)
        current_time = datetime.now()
        
        # Should be within 5 seconds of current time
        time_diff = abs((current_time - parsed_time).total_seconds())
        self.assertLess(time_diff, 5)


if __name__ == '__main__':
    unittest.main()