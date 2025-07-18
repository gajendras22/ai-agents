from strands import Agent, tool
from strands_tools import calculator, current_time, python_repl, use_aws
import PyPDF2
import boto3
import random
import os
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Note: Data models were originally planned but the implementation uses dictionaries
# for tool return values instead of dataclasses for simplicity and flexibility

# Define a custom tool as a Python function using the @tool decorator
@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0

    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())

@tool
def pdf_processor(start_page: Optional[int] = None, file_path: str = "book.pdf") -> Dict:
    """
    Extract text from random pages in a PDF document.
    
    This tool randomly selects a starting page and reads 2-3 consecutive pages,
    handling edge cases for pages near the end of the document.
    
    Args:
        start_page: Optional specific page to start from (1-indexed)
        file_path: Path to the PDF file (defaults to "book.pdf")
        
    Returns:
        dict: {
            'pages_read': List[int] - page numbers that were read,
            'extracted_text': str - combined text from all pages,
            'total_pages': int - total number of pages in the PDF,
            'success': bool - whether the operation was successful,
            'error': str - error message if operation failed
        }
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                'pages_read': [],
                'extracted_text': '',
                'total_pages': 0,
                'success': False,
                'error': f"PDF file not found: {file_path}"
            }
        
        # Open and read the PDF file
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                if total_pages == 0:
                    return {
                        'pages_read': [],
                        'extracted_text': '',
                        'total_pages': 0,
                        'success': False,
                        'error': "PDF file contains no pages"
                    }
                
                # Determine starting page
                if start_page is None:
                    # Random page selection (1-indexed)
                    start_page = random.randint(1, total_pages)
                else:
                    # Validate provided start_page
                    if start_page < 1 or start_page > total_pages:
                        return {
                            'pages_read': [],
                            'extracted_text': '',
                            'total_pages': total_pages,
                            'success': False,
                            'error': f"Invalid start_page: {start_page}. Must be between 1 and {total_pages}"
                        }
                
                # Determine how many pages to read (2-3 pages)
                pages_to_read = min(3, total_pages - start_page + 1)  # Handle edge case near end
                if pages_to_read < 2 and start_page > 1:
                    # If we're at the very end, try to read at least 2 pages by going back
                    start_page = max(1, total_pages - 1)
                    pages_to_read = min(2, total_pages - start_page + 1)
                
                # Extract text from selected pages
                extracted_text = ""
                pages_read = []
                
                for i in range(pages_to_read):
                    page_num = start_page + i
                    if page_num <= total_pages:
                        try:
                            page = pdf_reader.pages[page_num - 1]  # Convert to 0-indexed
                            page_text = page.extract_text()
                            
                            if page_text.strip():  # Only add non-empty pages
                                extracted_text += f"\n--- Page {page_num} ---\n"
                                extracted_text += page_text.strip() + "\n"
                                pages_read.append(page_num)
                        except Exception as page_error:
                            # Skip corrupted pages but continue with others
                            continue
                
                if not extracted_text.strip():
                    return {
                        'pages_read': pages_read,
                        'extracted_text': '',
                        'total_pages': total_pages,
                        'success': False,
                        'error': "No readable text found in selected pages"
                    }
                
                return {
                    'pages_read': pages_read,
                    'extracted_text': extracted_text.strip(),
                    'total_pages': total_pages,
                    'success': True,
                    'error': None
                }
                
            except PyPDF2.errors.PdfReadError as pdf_error:
                return {
                    'pages_read': [],
                    'extracted_text': '',
                    'total_pages': 0,
                    'success': False,
                    'error': f"Corrupted or invalid PDF file: {str(pdf_error)}"
                }
            except Exception as read_error:
                return {
                    'pages_read': [],
                    'extracted_text': '',
                    'total_pages': 0,
                    'success': False,
                    'error': f"Error reading PDF file: {str(read_error)}"
                }
                
    except PermissionError:
        return {
            'pages_read': [],
            'extracted_text': '',
            'total_pages': 0,
            'success': False,
            'error': f"Permission denied accessing file: {file_path}"
        }
    except Exception as e:
        return {
            'pages_read': [],
            'extracted_text': '',
            'total_pages': 0,
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }

@tool
def content_analyzer(text_content: str, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", use_mock: bool = False) -> Dict:
    """
    Analyze text content to extract key themes and messages using AWS Bedrock.
    
    This tool sends extracted text to AWS Bedrock for intelligent analysis,
    identifying core themes, key messages, and providing context summaries.
    
    Args:
        text_content: Raw text from PDF pages to analyze
        model_id: AWS Bedrock model ID to use for analysis
        use_mock: If True, use mock analysis for testing without AWS access
        
    Returns:
        dict: {
            'core_themes': List[str] - main themes identified in the content,
            'key_message': str - primary message or insight from the content,
            'context_summary': str - brief summary of the overall context,
            'confidence_score': float - confidence in the analysis (0.0-1.0),
            'success': bool - whether the analysis was successful,
            'error': str - error message if analysis failed
        }
    """
    try:
        # Validate input
        if not text_content or not text_content.strip():
            return {
                'core_themes': [],
                'key_message': '',
                'context_summary': '',
                'confidence_score': 0.0,
                'success': False,
                'error': 'No text content provided for analysis'
            }
        
        # Check if we should use mock analysis (for testing without AWS access)
        if use_mock or os.environ.get("USE_MOCK_ANALYSIS", "").lower() == "true":
            return _mock_content_analysis(text_content)
        
        # Prepare the prompt for content analysis
        analysis_prompt = f"""
Please analyze the following text content and provide a structured analysis. Focus on extracting meaningful insights that would be suitable for social media content.

TEXT TO ANALYZE:
{text_content}

Please provide your analysis in the following JSON format:
{{
    "core_themes": ["theme1", "theme2", "theme3"],
    "key_message": "The main insight or message from this content",
    "context_summary": "A brief summary of what this content is about",
    "confidence_score": 0.85
}}

Guidelines:
- Identify 2-4 core themes that capture the essence of the content
- Extract one key message that would resonate with social media audiences
- Provide a concise context summary (2-3 sentences max)
- Assign a confidence score between 0.0 and 1.0 based on content clarity
- Focus on insights that would make engaging social media posts
- Ensure themes are specific and actionable
"""

        # Use AWS Bedrock via use_aws tool
        bedrock_params = {
            "modelId": model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ]
            })
        }
        
        # Get AWS region from environment or default
        aws_region = os.environ.get("AWS_REGION", "us-west-2")
        
        # Create a mock ToolUse object for use_aws
        tool_use = {
            "toolUseId": f"content_analysis_{datetime.now().timestamp()}",
            "input": {
                "service_name": "bedrock-runtime",
                "operation_name": "invoke_model",
                "parameters": bedrock_params,
                "region": aws_region,
                "label": "Content Analysis with AWS Bedrock"
            }
        }
        
        # Call AWS Bedrock through use_aws
        from strands_tools.use_aws import use_aws as aws_tool
        bedrock_response = aws_tool(tool_use)
        
        # log bedrock_response
        print(f"Bedrock Response: {bedrock_response.get('content')}")

        if bedrock_response["status"] != "success":
            # If AWS call fails, provide helpful error message and suggest mock mode
            error_content = bedrock_response.get('content', [{}])[0].get('text', 'Unknown error')
            
            # Check if it's an access denied error and suggest using mock mode
            if "AccessDeniedException" in error_content or "access" in error_content.lower():
                return {
                    'core_themes': [],
                    'key_message': '',
                    'context_summary': '',
                    'confidence_score': 0.0,
                    'success': False,
                    'error': f"AWS Bedrock access denied. Ensure you have proper AWS credentials and Bedrock model access. For testing, you can use mock mode by setting USE_MOCK_ANALYSIS=true or calling with use_mock=True. Original error: {error_content}"
                }
            else:
                return {
                    'core_themes': [],
                    'key_message': '',
                    'context_summary': '',
                    'confidence_score': 0.0,
                    'success': False,
                    'error': f"AWS Bedrock call failed: {error_content}"
                }
        
        # Parse the response using the improved parsing technique
        response_text = bedrock_response["content"][0]["text"]
        
        # Use the parsing technique from your terminal example
        import ast
        
        try:
            # Check if response starts with "Success: "
            if response_text.startswith("Success: "):
                # Step 1: Strip the 'Success: ' prefix
                text_with_data = response_text[len("Success: "):]
                # Step 2: Use ast.literal_eval to safely parse the outer dict
                outer_dict = ast.literal_eval(text_with_data)
                # Step 3: Extract JSON string from inside the nested structure
                claude_response = outer_dict['body']['content'][0]['text']
            else:
                # Direct response format
                claude_response = response_text
        except (ValueError, KeyError, SyntaxError, TypeError) as parse_error:
            # Fallback to original parsing method
            claude_response = response_text
        
        # Try to parse the analysis result from Claude's response
        try:
            # Look for JSON in Claude's response
            json_match = re.search(r'\{.*\}', claude_response, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
                
                return {
                    'core_themes': analysis_result.get('core_themes', []),
                    'key_message': analysis_result.get('key_message', ''),
                    'context_summary': analysis_result.get('context_summary', ''),
                    'confidence_score': float(analysis_result.get('confidence_score', 0.7)),
                    'success': True,
                    'error': None
                }
            else:
                # Fallback: extract information manually from the response
                return {
                    'core_themes': ['General content analysis'],
                    'key_message': claude_response[:200] + "..." if len(claude_response) > 200 else claude_response,
                    'context_summary': 'Content analysis completed but structured format not detected',
                    'confidence_score': 0.6,
                    'success': True,
                    'error': None
                }
                
        except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
            # Fallback response if parsing fails
            return {
                'core_themes': ['Content analysis'],
                'key_message': 'Analysis completed with parsing issues',
                'context_summary': f'Raw response: {claude_response[:100]}...',
                'confidence_score': 0.5,
                'success': True,
                'error': f'Response parsing issue: {str(parse_error)}'
            }
            
    except Exception as e:
        return {
            'core_themes': [],
            'key_message': '',
            'context_summary': '',
            'confidence_score': 0.0,
            'success': False,
            'error': f"Content analysis failed: {str(e)}"
        }

def _mock_content_analysis(text_content: str) -> Dict:
    """
    Mock content analysis for testing without AWS access.
    
    This function provides a basic analysis based on text patterns and keywords
    to simulate what AWS Bedrock would return.
    """
    try:
        # Basic keyword extraction for themes
        text_lower = text_content.lower()
        
        # Define theme keywords
        theme_keywords = {
            'communication': ['communication', 'listen', 'speak', 'talk', 'conversation', 'dialogue'],
            'technology': ['digital', 'technology', 'tech', 'online', 'internet', 'computer'],
            'relationships': ['relationship', 'connect', 'trust', 'friendship', 'bond', 'social'],
            'learning': ['learn', 'education', 'knowledge', 'skill', 'understand', 'study'],
            'business': ['business', 'work', 'professional', 'career', 'success', 'strategy'],
            'personal_growth': ['growth', 'development', 'improve', 'better', 'change', 'transform'],
            'creativity': ['creative', 'art', 'design', 'innovation', 'imagination', 'inspire']
        }
        
        # Find matching themes
        detected_themes = []
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme.replace('_', ' ').title())
        
        # If no themes detected, use generic ones
        if not detected_themes:
            detected_themes = ['General Content', 'Information Sharing']
        
        # Limit to 3-4 themes
        detected_themes = detected_themes[:4]
        
        # Extract key message (first meaningful sentence)
        sentences = [s.strip() for s in text_content.split('.') if len(s.strip()) > 20]
        key_message = sentences[0] if sentences else "Key insights from the analyzed content"
        
        # Create context summary (first 150 characters + themes)
        context_summary = f"Content focuses on {', '.join(detected_themes[:2]).lower()}. {text_content[:100].strip()}..."
        
        # Calculate confidence based on text length and theme detection
        confidence = min(0.9, 0.5 + (len(detected_themes) * 0.1) + (min(len(text_content), 500) / 1000))
        
        return {
            'core_themes': detected_themes,
            'key_message': key_message,
            'context_summary': context_summary,
            'confidence_score': round(confidence, 2),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'core_themes': ['Content Analysis'],
            'key_message': 'Mock analysis completed with basic processing',
            'context_summary': 'Basic text analysis performed',
            'confidence_score': 0.5,
            'success': True,
            'error': f'Mock analysis warning: {str(e)}'
        }

@tool
def post_generator(analysis_result: Dict, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", use_mock: bool = False) -> Dict:
    """
    Generate social media post from content analysis using AWS Bedrock.
    
    This tool creates engaging 280-character social media posts optimized for platforms
    like X (Twitter), ensuring core message preservation within character limits.
    
    Args:
        analysis_result: Output from content_analyzer containing themes and key message
        model_id: AWS Bedrock model ID to use for post generation
        use_mock: If True, use mock generation for testing without AWS access
        
    Returns:
        dict: {
            'content': str - the generated social media post (≤280 characters),
            'character_count': int - actual character count of the post,
            'themes_used': List[str] - themes incorporated into the post,
            'generation_timestamp': str - ISO timestamp of generation,
            'success': bool - whether the generation was successful,
            'error': str - error message if generation failed,
            'optimization_notes': str - notes about post optimization applied
        }
    """
    try:
        # Validate input
        if not analysis_result or not isinstance(analysis_result, dict):
            return {
                'content': '',
                'character_count': 0,
                'themes_used': [],
                'generation_timestamp': datetime.now().isoformat(),
                'success': False,
                'error': 'Invalid analysis_result provided. Expected dictionary from content_analyzer.',
                'optimization_notes': ''
            }
        
        # Extract required fields from analysis result
        core_themes = analysis_result.get('core_themes', [])
        key_message = analysis_result.get('key_message', '')
        context_summary = analysis_result.get('context_summary', '')
        
        if not key_message and not context_summary:
            return {
                'content': '',
                'character_count': 0,
                'themes_used': [],
                'generation_timestamp': datetime.now().isoformat(),
                'success': False,
                'error': 'No meaningful content found in analysis_result to generate post from.',
                'optimization_notes': ''
            }
        
        # Check if we should use mock generation (for testing without AWS access)
        if use_mock or os.environ.get("USE_MOCK_GENERATION", "").lower() == "true":
            return _mock_post_generation(analysis_result)
        
        # Prepare the prompt for post generation
        themes_text = ", ".join(core_themes) if core_themes else "general insights"
        
        post_generation_prompt = f"""
Create an engaging social media post for X (Twitter) based on the following content analysis.

CONTENT ANALYSIS:
- Core Themes: {themes_text}
- Key Message: {key_message}
- Context: {context_summary}

REQUIREMENTS:
- Maximum 280 characters (STRICT LIMIT)
- Engaging and shareable for social media
- Preserve the core message and meaning
- Use active voice and compelling language
- Include relevant hashtags if space allows
- Make it conversational and relatable
- Focus on the most impactful insight

OPTIMIZATION GUIDELINES:
- Start with a hook or compelling statement
- Use emojis sparingly but effectively
- Keep sentences concise and punchy
- End with a call to action or thought-provoking question when possible
- Prioritize clarity over cleverness

Please respond with ONLY the social media post text, nothing else. The post must be exactly 280 characters or fewer.
"""

        # Use AWS Bedrock via use_aws tool
        bedrock_params = {
            "modelId": model_id,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": post_generation_prompt
                    }
                ]
            })
        }
        
        # Get AWS region from environment or default
        aws_region = os.environ.get("AWS_REGION", "us-west-2")
        
        # Create a mock ToolUse object for use_aws
        tool_use = {
            "toolUseId": f"post_generation_{datetime.now().timestamp()}",
            "input": {
                "service_name": "bedrock-runtime",
                "operation_name": "invoke_model",
                "parameters": bedrock_params,
                "region": aws_region,
                "label": "Social Post Generation with AWS Bedrock"
            }
        }
        
        # Call AWS Bedrock through use_aws
        from strands_tools.use_aws import use_aws as aws_tool
        bedrock_response = aws_tool(tool_use)
        
        if bedrock_response["status"] != "success":
            # If AWS call fails, provide helpful error message and suggest mock mode
            error_content = bedrock_response.get('content', [{}])[0].get('text', 'Unknown error')
            
            # Check if it's an access denied error and suggest using mock mode
            if "AccessDeniedException" in error_content or "access" in error_content.lower():
                return {
                    'content': '',
                    'character_count': 0,
                    'themes_used': [],
                    'generation_timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': f"AWS Bedrock access denied. Ensure you have proper AWS credentials and Bedrock model access. For testing, you can use mock mode by setting USE_MOCK_GENERATION=true or calling with use_mock=True. Original error: {error_content}",
                    'optimization_notes': ''
                }
            else:
                return {
                    'content': '',
                    'character_count': 0,
                    'themes_used': [],
                    'generation_timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': f"AWS Bedrock call failed: {error_content}",
                    'optimization_notes': ''
                }
        
        # Parse the response using the same improved technique as content_analyzer
        response_text = bedrock_response["content"][0]["text"]
        
        # Use the same parsing technique from content_analyzer
        import ast
        
        try:
            # Check if response starts with "Success: "
            if response_text.startswith("Success: "):
                # Step 1: Strip the 'Success: ' prefix
                text_with_data = response_text[len("Success: "):]
                # Step 2: Use ast.literal_eval to safely parse the outer dict
                outer_dict = ast.literal_eval(text_with_data)
                # Step 3: Extract JSON string from inside the nested structure
                generated_post = outer_dict['body']['content'][0]['text'].strip()
            else:
                # Direct response format
                generated_post = response_text.strip()
        except (ValueError, KeyError, SyntaxError, TypeError) as parse_error:
            # Fallback to original parsing method
            generated_post = response_text.strip()
        
        # Clean up the generated post
        generated_post = _clean_and_validate_post(generated_post)
        
        # Character count validation and optimization
        char_count = len(generated_post)
        optimization_notes = []
        
        if char_count > 280:
            # Truncate and optimize if too long
            generated_post, optimization_notes = _optimize_post_length(generated_post, core_themes)
            char_count = len(generated_post)
        
        # Determine which themes were used in the final post
        themes_used = []
        post_lower = generated_post.lower()
        for theme in core_themes:
            if any(word.lower() in post_lower for word in theme.split()):
                themes_used.append(theme)
        
        return {
            'content': generated_post,
            'character_count': char_count,
            'themes_used': themes_used,
            'generation_timestamp': datetime.now().isoformat(),
            'success': True,
            'error': None,
            'optimization_notes': '; '.join(optimization_notes) if optimization_notes else 'Post generated within character limits'
        }
        
    except Exception as e:
        return {
            'content': '',
            'character_count': 0,
            'themes_used': [],
            'generation_timestamp': datetime.now().isoformat(),
            'success': False,
            'error': f"Post generation failed: {str(e)}",
            'optimization_notes': ''
        }

def _mock_post_generation(analysis_result: Dict) -> Dict:
    """
    Mock post generation for testing without AWS access.
    
    This function creates a basic social media post based on the analysis result
    to simulate what AWS Bedrock would generate.
    """
    try:
        core_themes = analysis_result.get('core_themes', [])
        key_message = analysis_result.get('key_message', '')
        context_summary = analysis_result.get('context_summary', '')
        
        # Create a basic post structure
        if key_message:
            # Use the key message as the base
            base_content = key_message
        elif context_summary:
            # Fall back to context summary
            base_content = context_summary
        else:
            base_content = "Interesting insights from recent content analysis"
        
        # Truncate to fit within reasonable limits for further processing
        if len(base_content) > 200:
            base_content = base_content[:197] + "..."
        
        # Add a simple call to action or engagement element
        engagement_endings = [
            " What do you think?",
            " Thoughts?",
            " 💭",
            " Worth considering!",
            " Food for thought.",
            " 🤔"
        ]
        
        # Choose ending based on available space
        for ending in engagement_endings:
            if len(base_content) + len(ending) <= 280:
                base_content += ending
                break
        
        # Add hashtag if themes are available and space permits
        if core_themes and len(base_content) < 260:
            # Create a simple hashtag from the first theme
            theme_hashtag = f" #{core_themes[0].replace(' ', '').replace('-', '')}"
            if len(base_content) + len(theme_hashtag) <= 280:
                base_content += theme_hashtag
        
        # Final validation and cleanup
        final_post = _clean_and_validate_post(base_content)
        char_count = len(final_post)
        
        # Determine themes used
        themes_used = []
        post_lower = final_post.lower()
        for theme in core_themes:
            if any(word.lower() in post_lower for word in theme.split()):
                themes_used.append(theme)
        
        return {
            'content': final_post,
            'character_count': char_count,
            'themes_used': themes_used,
            'generation_timestamp': datetime.now().isoformat(),
            'success': True,
            'error': None,
            'optimization_notes': 'Mock generation - basic post structure with engagement elements'
        }
        
    except Exception as e:
        return {
            'content': 'Mock post generation completed with basic processing.',
            'character_count': 52,
            'themes_used': [],
            'generation_timestamp': datetime.now().isoformat(),
            'success': True,
            'error': None,
            'optimization_notes': f'Mock generation warning: {str(e)}'
        }

def _clean_and_validate_post(post_content: str) -> str:
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

def _optimize_post_length(post_content: str, themes: List[str]) -> tuple:
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

@tool
def generate_social_post(file_path: str = "book.pdf", start_page: Optional[int] = None, use_mock: bool = False) -> Dict:
    """
    Main workflow orchestrator that coordinates the entire PDF-to-social-post pipeline.
    
    This tool orchestrates the complete workflow:
    1. Extract text from random PDF pages
    2. Analyze content for themes and key messages
    3. Generate optimized social media post
    
    Args:
        file_path: Path to the PDF file (defaults to "book.pdf")
        start_page: Optional specific page to start from (1-indexed)
        use_mock: If True, use mock analysis and generation for testing
        
    Returns:
        dict: {
            'social_post': str - the final generated social media post,
            'character_count': int - character count of the post,
            'pages_processed': List[int] - pages that were read from PDF,
            'themes_identified': List[str] - core themes found in content,
            'confidence_score': float - analysis confidence (0.0-1.0),
            'processing_summary': str - summary of the workflow execution,
            'success': bool - whether the entire workflow was successful,
            'error': str - error message if workflow failed,
            'detailed_results': dict - detailed results from each step
        }
    """
    try:
        workflow_results = {
            'pdf_processing': None,
            'content_analysis': None,
            'post_generation': None
        }
        
        print("🔄 Starting PDF-to-Social-Post workflow...")
        
        # Step 1: Process PDF and extract text
        print("📄 Step 1: Processing PDF and extracting text...")
        pdf_result = pdf_processor(start_page=start_page, file_path=file_path)
        workflow_results['pdf_processing'] = pdf_result
        
        if not pdf_result.get('success', False):
            return {
                'social_post': '',
                'character_count': 0,
                'pages_processed': [],
                'themes_identified': [],
                'confidence_score': 0.0,
                'processing_summary': f"Workflow failed at PDF processing: {pdf_result.get('error', 'Unknown error')}",
                'success': False,
                'error': pdf_result.get('error', 'PDF processing failed'),
                'detailed_results': workflow_results
            }
        
        extracted_text = pdf_result.get('extracted_text', '')
        pages_read = pdf_result.get('pages_read', [])
        
        print(f"✅ Successfully extracted text from pages: {pages_read}")
        print(f"📝 Text length: {len(extracted_text)} characters")
        
        # Step 2: Analyze content for themes and key messages
        print("🧠 Step 2: Analyzing content for themes and insights...")
        analysis_result = content_analyzer(text_content=extracted_text, use_mock=use_mock)
        workflow_results['content_analysis'] = analysis_result
        
        if not analysis_result.get('success', False):
            return {
                'social_post': '',
                'character_count': 0,
                'pages_processed': pages_read,
                'themes_identified': [],
                'confidence_score': 0.0,
                'processing_summary': f"Workflow failed at content analysis: {analysis_result.get('error', 'Unknown error')}",
                'success': False,
                'error': analysis_result.get('error', 'Content analysis failed'),
                'detailed_results': workflow_results
            }
        
        core_themes = analysis_result.get('core_themes', [])
        key_message = analysis_result.get('key_message', '')
        confidence_score = analysis_result.get('confidence_score', 0.0)
        
        print(f"✅ Identified themes: {', '.join(core_themes)}")
        print(f"🎯 Key message: {key_message[:100]}{'...' if len(key_message) > 100 else ''}")
        print(f"📊 Confidence score: {confidence_score}")
        
        # Step 3: Generate social media post
        print("📱 Step 3: Generating social media post...")
        post_result = post_generator(analysis_result=analysis_result, use_mock=use_mock)
        workflow_results['post_generation'] = post_result
        
        if not post_result.get('success', False):
            return {
                'social_post': '',
                'character_count': 0,
                'pages_processed': pages_read,
                'themes_identified': core_themes,
                'confidence_score': confidence_score,
                'processing_summary': f"Workflow failed at post generation: {post_result.get('error', 'Unknown error')}",
                'success': False,
                'error': post_result.get('error', 'Post generation failed'),
                'detailed_results': workflow_results
            }
        
        final_post = post_result.get('content', '')
        char_count = post_result.get('character_count', 0)
        themes_used = post_result.get('themes_used', [])
        
        print(f"✅ Generated post ({char_count} characters):")
        print(f"📝 {final_post}")
        print(f"🏷️  Themes used: {', '.join(themes_used)}")
        
        # Create processing summary
        processing_summary = f"Successfully processed {len(pages_read)} pages from {file_path}, identified {len(core_themes)} themes, and generated a {char_count}-character social media post."
        
        print("🎉 Workflow completed successfully!")
        
        return {
            'social_post': final_post,
            'character_count': char_count,
            'pages_processed': pages_read,
            'themes_identified': core_themes,
            'confidence_score': confidence_score,
            'processing_summary': processing_summary,
            'success': True,
            'error': None,
            'detailed_results': workflow_results
        }
        
    except Exception as e:
        error_message = f"Workflow orchestrator failed: {str(e)}"
        print(f"❌ {error_message}")
        
        return {
            'social_post': '',
            'character_count': 0,
            'pages_processed': [],
            'themes_identified': [],
            'confidence_score': 0.0,
            'processing_summary': error_message,
            'success': False,
            'error': error_message,
            'detailed_results': workflow_results
        }

# Create an agent with tools from the strands-tools example tools package
# as well as our custom tools
agent = Agent(tools=[calculator, current_time, python_repl, use_aws, letter_counter, pdf_processor, content_analyzer, post_generator, generate_social_post], model="anthropic.claude-3-sonnet-20240229-v1:0")

# Example usage and testing
if __name__ == "__main__":
    
    message = """
    Please analyze this text using the content_analyzer tool and generate a post using the post_generator tool:
    "One mistake that some teams make is delaying the demo of the product to the customer until development is complete. That’s a big risk. It’s better to keep doing demos in a phased manner with the client and get feedback. This helps in easier course correction. And helps the team to maintain pace in providing deliveries. There is also that sense of accomplishment with each incremental demo that helps boost the team’s confidence. Plan to have these demos at significant milestones throughout the development phase of projects. Do not wait for permission or for your client to request a demo. Be proactive and schedule a demo with the client and if possible, the actual end users of the system"
    """
    agent(message)