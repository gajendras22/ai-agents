"""
Utility functions for the agent refactoring project.

This module contains shared utility functions used across multiple modules
for post cleaning, validation, response parsing, and input validation.
"""

import re
import ast
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def clean_and_validate_post(post_content: str) -> str:
    """
    Clean and validate social media post content.
    
    Args:
        post_content: Raw post content to clean
        
    Returns:
        str: Cleaned post content
    """
    # Remove extra whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', post_content.strip())
    
    # Remove any quotes that might wrap the entire post
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()
    
    # Remove any prefixes that might have been added by the LLM
    prefixes_to_remove = [
        "Here's a social media post:",
        "Social media post:",
        "Tweet:",
        "Post:",
        "Here's the post:",
        "Generated post:"
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned


def optimize_post_length(post_content: str, themes: List[str]) -> Tuple[str, List[str]]:
    """
    Optimize post length to fit within 280 characters while preserving meaning.
    
    Args:
        post_content: Original post content that's too long
        themes: Core themes to preserve if possible
        
    Returns:
        tuple: (optimized_post, optimization_notes)
    """
    optimization_notes = []
    
    # If it's way too long, truncate aggressively first
    if len(post_content) > 350:
        post_content = post_content[:277] + "..."
        optimization_notes.append("Aggressive truncation applied")
    
    # Try to optimize by removing less important elements
    optimizations = [
        # Remove extra punctuation
        (r'\.{2,}', '...', "Multiple periods replaced with ellipsis"),
        (r'!{2,}', '!', "Multiple exclamation marks reduced"),
        (r'\?{2,}', '?', "Multiple question marks reduced"),
        
        # Shorten common phrases
        (r'\bthat is\b', "that's", "Contracted 'that is'"),
        (r'\bdo not\b', "don't", "Contracted 'do not'"),
        (r'\bcannot\b', "can't", "Contracted 'cannot'"),
        (r'\bwill not\b', "won't", "Contracted 'will not'"),
        (r'\byou are\b', "you're", "Contracted 'you are'"),
        (r'\bit is\b', "it's", "Contracted 'it is'"),
        
        # Remove filler words
        (r'\bvery\s+', '', "Removed 'very'"),
        (r'\breally\s+', '', "Removed 'really'"),
        (r'\bquite\s+', '', "Removed 'quite'"),
        (r'\bactually\s+', '', "Removed 'actually'"),
    ]
    
    for pattern, replacement, note in optimizations:
        if len(post_content) <= 280:
            break
        
        old_content = post_content
        post_content = re.sub(pattern, replacement, post_content, flags=re.IGNORECASE)
        
        if post_content != old_content:
            optimization_notes.append(note)
    
    # If still too long, truncate at word boundary
    if len(post_content) > 280:
        # Find the last complete word that fits
        truncated = post_content[:277]
        last_space = truncated.rfind(' ')
        
        if last_space > 200:  # Only truncate at word boundary if it's reasonable
            post_content = truncated[:last_space] + "..."
            optimization_notes.append("Truncated at word boundary")
        else:
            post_content = truncated + "..."
            optimization_notes.append("Hard truncation applied")
    
    return post_content, optimization_notes


def parse_bedrock_response(response_text: str) -> str:
    """
    Parse AWS Bedrock response text using improved parsing technique.
    
    Args:
        response_text: Raw response text from AWS Bedrock
        
    Returns:
        str: Parsed response content
    """
    try:
        # Check if response starts with "Success: "
        if response_text.startswith("Success: "):
            # Step 1: Strip the 'Success: ' prefix
            text_with_data = response_text[len("Success: "):]
            # Step 2: Use ast.literal_eval to safely parse the outer dict
            outer_dict = ast.literal_eval(text_with_data)
            # Step 3: Extract JSON string from inside the nested structure
            parsed_response = outer_dict['body']['content'][0]['text']
        else:
            # Direct response format
            parsed_response = response_text
        
        return parsed_response.strip()
        
    except (ValueError, KeyError, SyntaxError, TypeError):
        # Fallback to original parsing method
        return response_text.strip()


def validate_analysis_input(analysis_result: Dict) -> bool:
    """
    Validate analysis result input for post generation.
    
    Args:
        analysis_result: Dictionary containing analysis results
        
    Returns:
        bool: True if analysis_result is valid, False otherwise
    """
    if not analysis_result or not isinstance(analysis_result, dict):
        return False
    
    # Check for required fields
    key_message = analysis_result.get('key_message', '')
    context_summary = analysis_result.get('context_summary', '')
    
    # At least one of these should have meaningful content
    return bool(key_message.strip() or context_summary.strip())


def validate_pdf_input(file_path: str, start_page: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate PDF processing input parameters.
    
    Args:
        file_path: Path to the PDF file
        start_page: Optional starting page number
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file_path or not isinstance(file_path, str):
        return False, "Invalid file_path: must be a non-empty string"
    
    if start_page is not None:
        if not isinstance(start_page, int) or start_page < 1:
            return False, "Invalid start_page: must be a positive integer"
    
    return True, ""


def create_error_response(error_message: str, **additional_fields) -> Dict:
    """
    Create a standardized error response dictionary.
    
    Args:
        error_message: The error message to include
        **additional_fields: Additional fields to include in the response
        
    Returns:
        dict: Standardized error response
    """
    error_response = {
        'success': False,
        'error': error_message,
        **additional_fields
    }
    
    return error_response


def create_success_response(**fields) -> Dict:
    """
    Create a standardized success response dictionary.
    
    Args:
        **fields: Fields to include in the success response
        
    Returns:
        dict: Standardized success response
    """
    success_response = {
        'success': True,
        'error': None,
        **fields
    }
    
    return success_response


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object from text content.
    
    Args:
        text: Text content that may contain JSON
        
    Returns:
        dict or None: Parsed JSON object if found, None otherwise
    """
    try:
        # Look for JSON in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except (json.JSONDecodeError, AttributeError):
        return None


def sanitize_text_content(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text content by removing excessive whitespace and optionally truncating.
    
    Args:
        text: Text content to sanitize
        max_length: Optional maximum length to truncate to
        
    Returns:
        str: Sanitized text content
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if max_length is specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length-3] + "..."
    
    return sanitized


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp in ISO format
    """
    return datetime.now().isoformat()