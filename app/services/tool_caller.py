import json
import logging
import random
from typing import List, Dict, Any
import cloudscraper

logger = logging.getLogger(__name__)

class ToolCaller:
    def __init__(self):
        self.mcp_url = "https://mcp.exa.ai/mcp"
        self.mcp_params = {
            "profile": "joyous-gull-NeZ2gW",
            "api_key": "fe5676be-931d-42e1-b5c9-90e94dce45ae"
        }
        self.scraper = cloudscraper.create_scraper()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        # 从情报中提取的工具定义
        return [
            {"name":"resolve-library-id","title":"Resolve Context7 Library ID","description":"...","inputSchema":{}},
            {"name":"get-library-docs","title":"Get Library Docs","description":"...","inputSchema":{}},
            {"name":"web_search_exa","description":"...","inputSchema":{}},
            {"name":"get_code_context_exa","description":"...","inputSchema":{}}
        ]

    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 此函数现在返回一个单一的用户消息，其中包含所有工具的结果
        all_results_content = "Tool results:\n"
        
        for call in tool_calls:
            function_call = call.get("function", {})
            tool_name = function_call.get("name")
            tool_args_str = function_call.get("arguments", "{}")
            tool_call_id = call.get("id")

            logger.info(f"执行工具调用: {tool_name} with args {tool_args_str}")
            
            try:
                arguments = json.loads(tool_args_str)
                payload = {
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                    "jsonrpc": "2.0",
                    "id": random.randint(1, 100)
                }
                
                response = self.scraper.post(self.mcp_url, params=self.mcp_params, json=payload)
                response.raise_for_status()
                
                result_content = "No content returned."
                for line in response.iter_lines():
                    if line.startswith(b"data:"):
                        content_str = line[len(b"data:"):].strip().decode('utf-8', errors='ignore')
                        if content_str == "[DONE]":
                            break
                        try:
                            data = json.loads(content_str)
                            if "result" in data and "content" in data["result"]:
                                # 将结果格式化为更易读的字符串
                                result_content = json.dumps(data["result"]["content"], ensure_ascii=False, indent=2)
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"MCP tool: 无法解析 SSE 数据块: {content_str}")
                            continue
                
                all_results_content += f"\n--- Result for {tool_name} (call_id: {tool_call_id}) ---\n{result_content}\n"

            except Exception as e:
                logger.error(f"工具调用失败: {e}", exc_info=True)
                error_str = f'{{"error": "Tool call failed", "details": "{str(e)}"}}'
                all_results_content += f"\n--- Error for {tool_name} (call_id: {tool_call_id}) ---\n{error_str}\n"

        # 返回一个单一的用户消息，而不是多个 tool 角色的消息
        return [{"role": "user", "content": all_results_content}]
