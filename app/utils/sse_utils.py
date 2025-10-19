import json
import time
from typing import Dict, Any, Optional, List

DONE_CHUNK = b"data: [DONE]\n\n"

def create_sse_data(data: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode('utf-8')

def create_chat_completion_chunk(
    request_id: str,
    model: str,
    content: str,
    finish_reason: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason
            }
        ]
    }

def create_tool_call_chunk(
    request_id: str,
    model: str,
    tool_call_id: str,
    function_name: str,
    function_arguments: str = "",
    finish_reason: Optional[str] = None
) -> Dict[str, Any]:
    """创建工具调用的流式响应块"""
    delta = {}

    if function_arguments:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_arguments
                }
            }
        ]
    else:
        # 工具调用开始时的初始块
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": ""
                }
            }
        ]

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
