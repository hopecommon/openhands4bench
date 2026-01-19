"""
Convert OpenHands internal Message objects to benchmark results format.

This module provides utilities to transform conversation messages into the format
expected by benchmark evaluation scripts, ensuring strict alignment with reference
output files.
"""

from typing import Any

from openhands.core.message import Message, TextContent, ImageContent


def _content_to_string(content: str | list[TextContent | ImageContent] | None) -> str | None:
    """
    Convert message content to string format.
    
    Handles:
    - Plain strings
    - List of TextContent/ImageContent objects
    - None values
    
    For images, uses the 'url' field if available.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, TextContent):
                parts.append(item.text)
            elif isinstance(item, ImageContent):
                # Use URL for images (benchmark format compatibility)
                parts.append(item.url or '')
            elif isinstance(item, dict):
                # Fallback for dict-like content
                if 'text' in item:
                    parts.append(str(item['text']))
                elif 'url' in item:
                    parts.append(str(item['url']))
        return '\n'.join(parts) if parts else None
    return str(content)


def _tool_calls_to_results(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """
    Convert tool calls to benchmark results format.
    
    Key transformations:
    - Ensure 'index' is always -1 (benchmark convention)
    - Preserve all required fields: id, function, type
    - Keep 'arguments' as JSON string
    """
    if not tool_calls:
        return None
    
    result = []
    for tc in tool_calls:
        # Handle both dict and object forms
        if hasattr(tc, '__dict__'):
            tc_dict = {
                'id': getattr(tc, 'id', None),
                'function': getattr(tc, 'function', {}),
                'type': getattr(tc, 'type', 'function'),
            }
        else:
            tc_dict = dict(tc)
        
        # Convert function object to dict if needed
        if 'function' in tc_dict:
            func = tc_dict['function']
            # If function is an object, convert to dict
            if hasattr(func, '__dict__') and not isinstance(func, dict):
                func = {
                    'name': getattr(func, 'name', None),
                    'arguments': getattr(func, 'arguments', '{}'),
                }
                tc_dict['function'] = func
            
            # Ensure function.arguments is a string (not dict)
            if isinstance(func, dict) and 'arguments' in func:
                if not isinstance(func['arguments'], str):
                    import json
                    func['arguments'] = json.dumps(func['arguments'], ensure_ascii=False)
        
        # Benchmark format requires index=-1
        tc_dict['index'] = -1
        result.append(tc_dict)
    
    return result


def messages_to_results_format(messages: list[Message]) -> list[dict[str, Any]]:
    """
    Transform OpenHands Message objects to benchmark results format.
    
    Format requirements (from reference files):
    
    1. System/User messages:
       {"role": "system", "content": "..."}
       {"role": "user", "content": "..."}
    
    2. Assistant messages (with tool_calls):
       {
           "content": null,  # null when tool_calls present
           "refusal": null,
           "role": "assistant",
           "annotations": null,
           "audio": null,
           "function_call": null,
           "tool_calls": [...],
           "reasoning_content": "..."  # if available
       }
    
    3. Tool messages:
       {
           "role": "tool",
           "tool_call_id": "call_xxx",
           "name": "tool_name",
           "content": "..."
       }
    
    Args:
        messages: List of OpenHands Message objects
    
    Returns:
        List of dicts in benchmark format
    """
    result = []
    
    for msg in messages:
        role = msg.role
        
        if role in ('system', 'user'):
            # Simple format for system/user
            result.append({
                'role': role,
                'content': _content_to_string(msg.content),
            })
        
        elif role == 'assistant':
            # Full format with all fields
            tool_calls = _tool_calls_to_results(msg.tool_calls)
            
            # IMPORTANT: Always preserve content, even when tool_calls are present!
            # The assistant can explain what it's doing AND call tools simultaneously.
            # Setting content=null would lose valuable information.
            content_val = _content_to_string(msg.content)
            
            # Extract reasoning from multiple possible sources
            reasoning = None
            if hasattr(msg, 'reasoning') and msg.reasoning:
                reasoning = msg.reasoning
            elif hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                reasoning = msg.reasoning_content
            
            result.append({
                'content': content_val,
                'refusal': None,
                'role': 'assistant',
                'annotations': None,
                'audio': None,
                'function_call': None,
                'tool_calls': tool_calls,
                'reasoning_content': reasoning,
            })
        
        elif role == 'tool':
            # Tool response format
            result.append({
                'role': 'tool',
                'tool_call_id': getattr(msg, 'tool_call_id', None),
                'name': getattr(msg, 'name', None),
                'content': _content_to_string(msg.content),
            })
    
    return result
