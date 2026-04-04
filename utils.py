import json
import logging
from api import call_deepseek_wanqing as call_llm
import config

def get_json(json_string):
    js_result = json.loads(json_string)
    t = 0
    while type(js_result) == str and t < 10:
        t += 1
        js_result = json.loads(js_result)
    return js_result

def fix_and_parse_json(json_string: str, logger: logging.Logger, try_to_fix: bool = True):
    """
    Parses a JSON string. If parsing fails, optionally tries to fix it using an LLM.

    Args:
        json_string: The string to parse.
        logger: The logger to use for warnings and errors.
        try_to_fix: Whether to try to fix the JSON using an LLM if parsing fails.

    Returns:
        The parsed JSON object, or None if parsing and fixing fail.
    """
    try:
        # Handle potential markdown code block
        if json_string.startswith("```json"):
            json_string = json_string[7:-3].strip()
        elif json_string.startswith("```"):
            json_string = json_string[3:-3].strip()
        return get_json(json_string)
    except json.JSONDecodeError:
        if not try_to_fix:
            logger.error(f"Failed to parse JSON: {json_string}")
            return None
        
        logger.warning(f"Malformed JSON detected: {json_string}. Attempting to fix with LLM.")
        fixer_prompt = f"The following string is a malformed JSON. Correct it and return only the valid JSON object:\n\n{json_string}"
        
        try:
            # In A_LocalizeAgent.py, it uses call_summarizer_llm, others use call_deepseek_wanqing
            # We need a unified way to call the LLM. For now, we'll use call_deepseek_wanqing
            # as it seems to be the more common one.
            # You might need to adjust this part based on your specific LLM calling conventions.
            from api import call_deepseek_wanqing as call_llm 
            
            response = call_llm([{"role": "user", "content": fixer_prompt}])
            corrected_str = response.get("content", "").strip()

            if corrected_str.startswith("```json"):
                corrected_str = corrected_str[7:-3].strip()
            elif corrected_str.startswith("```"):
                corrected_str = corrected_str[3:-3].strip()
                
            return get_json(corrected_str)
        except Exception as e:
            logger.error(f"Failed to fix and parse JSON after LLM attempt: {e}")
            return None