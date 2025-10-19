import json
import time
import logging
import uuid
import random
import cloudscraper
from typing import Dict, Any, AsyncGenerator, List

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.config import settings
from app.providers.base_provider import BaseProvider
# 移除了不再使用的 SessionManager
# from app.services.session_manager import SessionManager
from app.utils.sse_utils import create_sse_data, create_chat_completion_chunk, create_tool_call_chunk, DONE_CHUNK

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class SmitheryProvider(BaseProvider):
    def __init__(self):
        # self.session_manager = SessionManager() # 移除会话管理器
        self.scraper = cloudscraper.create_scraper()
        self.cookie_index = 0

    def _get_cookie(self) -> str:
        """从配置中轮换获取一个格式正确的 Cookie 字符串。"""
        auth_cookie_obj = settings.AUTH_COOKIES[self.cookie_index]
        self.cookie_index = (self.cookie_index + 1) % len(settings.AUTH_COOKIES)
        return auth_cookie_obj.header_cookie_string

    def _convert_messages_to_smithery_format(self, openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将客户端发来的 OpenAI 格式消息列表转换为 Smithery.ai 后端所需的格式。
        支持文本和图像内容。

        OpenAI 格式示例:
        - 纯文本: {"role": "user", "content": "你好"}
        - 多模态: {"role": "user", "content": [{"type": "text", "text": "你好"}, {"type": "image_url", "image_url": {"url": "data:image/..."}}]}

        Smithery 格式:
        {"role": "user", "parts": [{"type": "text", "text": "你好"}, {"type": "file", "mediaType": "image/png", "url": "data:image/..."}]}
        """
        smithery_messages = []
        for msg in openai_messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls", [])

            if not role:
                continue

            parts = []

            # 处理工具调用消息 (assistant role with tool_calls)
            if role == "assistant" and tool_calls:
                # 添加助手响应文本 (如果有)
                if content and content.strip():
                    parts.append({"type": "text", "text": content})

                # 添加工具调用
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function = tool_call.get("function", {})
                        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
                        function_name = function.get("name", "")
                        function_args = function.get("arguments", "{}")

                        if function_name:
                            try:
                                # 解析函数参数
                                args_dict = json.loads(function_args) if isinstance(function_args, str) else function_args
                                parts.append({
                                    "type": f"tool-{function_name}",
                                    "toolCallId": tool_call_id,
                                    "state": "output-available",
                                    "input": args_dict
                                })
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析工具调用参数: {function_args}")

            # 处理工具响应消息 (tool role)
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if content and tool_call_id:
                    # 工具响应消息转换为特定格式
                    parts.append({
                        "type": "tool-response",
                        "toolCallId": tool_call_id,
                        "content": content
                    })

            # 处理普通内容消息
            elif content:
                # 处理不同的内容格式
                if isinstance(content, str):
                    # 简单文本消息
                    if content.strip():
                        parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # 多模态消息 (文本 + 图像)
                    for item in content:
                        if not isinstance(item, dict):
                            continue

                        item_type = item.get("type")

                        if item_type == "text":
                            text_content = item.get("text", "")
                            if text_content.strip():
                                parts.append({"type": "text", "text": text_content})

                        elif item_type == "image_url":
                            image_url_data = item.get("image_url", {})
                            image_url = image_url_data.get("url", "")

                            if image_url.startswith("data:image/"):
                                # 从 data URL 中提取媒体类型
                                media_type = self._extract_media_type_from_data_url(image_url)
                                file_extension = media_type.split("/")[1] if "/" in media_type else "png"

                                parts.append({
                                    "type": "file",
                                    "mediaType": media_type,
                                    "filename": f"image.{file_extension}",
                                    "url": image_url
                                })

            # 只有在有有效内容时才添加消息
            if parts:
                smithery_messages.append({
                    "role": role,
                    "parts": parts,
                    "id": f"msg-{uuid.uuid4().hex[:16]}"
                })

        return smithery_messages

    def _extract_media_type_from_data_url(self, data_url: str) -> str:
        """
        从 data URL 中提取媒体类型。
        例如: "data:image/png;base64,..." -> "image/png"
        """
        try:
            if data_url.startswith("data:"):
                # 格式: data:mediatype;base64,data
                parts = data_url.split(";", 1)
                if len(parts) >= 1:
                    media_type = parts[0][5:]  # 移除 "data:" 前缀
                    return media_type if media_type else "image/png"
        except Exception:
            pass
        return "image/png"  # 默认值

    async def chat_completion(self, request_data: Dict[str, Any]) -> StreamingResponse:
        """
        处理聊天补全请求。
        此实现为无状态模式，完全依赖客户端发送的完整对话历史。
        """
        
        # 1. 直接从客户端请求中获取完整的消息历史和工具定义
        messages_from_client = request_data.get("messages", [])
        tools_from_client = request_data.get("tools", [])

        # 2. 将其转换为 Smithery.ai 后端所需的格式
        smithery_formatted_messages = self._convert_messages_to_smithery_format(messages_from_client)

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            request_id = f"chatcmpl-{uuid.uuid4()}"
            model = request_data.get("model", "claude-haiku-4.5")
            
            try:
                # 3. 使用转换后的消息列表和工具定义准备请求体
                payload = self._prepare_payload(model, smithery_formatted_messages, tools_from_client)
                headers = self._prepare_headers()
                
                logger.info("===================== [REQUEST TO SMITHERY (Stateless)] =====================")
                logger.info(f"URL: POST {settings.CHAT_API_URL}")
                logger.info(f"PAYLOAD:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
                logger.info("=====================================================================================")

                # 使用 cloudscraper 发送请求
                response = self.scraper.post(
                    settings.CHAT_API_URL, 
                    headers=headers, 
                    json=payload, 
                    stream=True, 
                    timeout=settings.API_REQUEST_TIMEOUT
                )

                if response.status_code != 200:
                    logger.error("==================== [RESPONSE FROM SMITHERY (ERROR)] ===================")
                    logger.error(f"STATUS CODE: {response.status_code}")
                    logger.error(f"RESPONSE BODY:\n{response.text}")
                    logger.error("=================================================================")
                
                response.raise_for_status()

                # 4. 流式处理返回的数据，支持文本和工具调用
                for line in response.iter_lines():
                    if line.startswith(b"data:"):
                        content = line[len(b"data:"):].strip()
                        if content == b"[DONE]":
                            break
                        try:
                            data = json.loads(content)
                            response_type = data.get("type", "")

                            # 处理文本内容
                            if response_type == "text-delta":
                                delta_content = data.get("delta", "")
                                chunk = create_chat_completion_chunk(request_id, model, delta_content)
                                yield create_sse_data(chunk)

                            # 处理步骤开始/结束
                            elif response_type in ["step-start", "start-step"]:
                                # Smithery 发送 step-start 表示开始工具调用步骤
                                continue
                            elif response_type in ["finish-step"]:
                                # 工具执行步骤完成
                                continue

                            # 处理文本开始/结束
                            elif response_type in ["text-start", "text-end"]:
                                # 文本块的开始和结束标记
                                continue

                            # 处理工具输入开始
                            elif response_type == "tool-input-start":
                                tool_call_id = data.get("toolCallId", f"call_{uuid.uuid4().hex[:8]}")
                                tool_name = data.get("toolName", "unknown_tool")

                                # 发送工具调用开始的块
                                chunk = create_tool_call_chunk(request_id, model, tool_call_id, tool_name)
                                yield create_sse_data(chunk)

                            # 处理工具输入增量数据（构建参数）
                            elif response_type == "tool-input-delta":
                                # 这些是构建工具参数的增量数据，我们可以忽略，因为完整参数会在 tool-input-available 中提供
                                continue

                            # 处理工具输入完成（参数已完整）
                            elif response_type == "tool-input-available":
                                tool_call_id = data.get("toolCallId", f"call_{uuid.uuid4().hex[:8]}")
                                tool_name = data.get("toolName", "unknown_tool")
                                tool_input = data.get("input", {})

                                # 发送工具调用参数
                                if tool_input:
                                    chunk = create_tool_call_chunk(
                                        request_id, model, tool_call_id, tool_name,
                                        json.dumps(tool_input)
                                    )
                                    yield create_sse_data(chunk)

                            # 处理传统的工具调用（保持向后兼容）
                            elif response_type.startswith("tool-"):
                                tool_name = response_type[5:]  # 移除 "tool-" 前缀
                                tool_call_id = data.get("toolCallId", f"call_{uuid.uuid4().hex[:8]}")
                                tool_input = data.get("input", {})
                                tool_state = data.get("state", "")

                                # 发送工具调用开始的块
                                if tool_state in ["pending", "running"]:
                                    chunk = create_tool_call_chunk(
                                        request_id, model, tool_call_id, tool_name
                                    )
                                    yield create_sse_data(chunk)

                                    # 发送工具调用参数
                                    if tool_input:
                                        chunk = create_tool_call_chunk(
                                            request_id, model, tool_call_id, tool_name,
                                            json.dumps(tool_input)
                                        )
                                        yield create_sse_data(chunk)

                                # 工具调用完成时，可以发送结果作为文本
                                elif tool_state == "output-available":
                                    tool_output = data.get("output", {})
                                    if tool_output:
                                        # 将工具输出作为助手消息返回
                                        output_text = self._format_tool_output(tool_output)
                                        if output_text:
                                            chunk = create_chat_completion_chunk(request_id, model, output_text)
                                            yield create_sse_data(chunk)

                            # 处理开始/结束标记
                            elif response_type in ["start", "finish"]:
                                # 整个会话的开始和结束标记
                                continue

                            # 处理其他类型的响应
                            else:
                                logger.debug(f"收到未处理的响应类型: {response_type}")

                        except json.JSONDecodeError:
                            if content:
                                logger.warning(f"无法解析 SSE 数据块: {content}")
                            continue
                
                # 5. 无状态模式下，无需保存任何会话，直接发送结束标志
                final_chunk = create_chat_completion_chunk(request_id, model, "", "stop")
                yield create_sse_data(final_chunk)
                yield DONE_CHUNK

            except Exception as e:
                logger.error(f"处理流时发生错误: {e}", exc_info=True)
                error_message = f"内部服务器错误: {str(e)}"
                error_chunk = create_chat_completion_chunk(request_id, model, error_message, "stop")
                yield create_sse_data(error_chunk)
                yield DONE_CHUNK

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def _prepare_headers(self) -> Dict[str, str]:
        # 包含我们之前分析出的所有必要请求头
        return {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "Cookie": self._get_cookie(),
            "Origin": "https://smithery.ai",
            "Referer": "https://smithery.ai/playground",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "priority": "u=1, i",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-posthog-distinct-id": "5905f6b4-d74f-46b4-9b4f-9dbbccb29bee",
            "x-posthog-session-id": "0199f71f-8c42-7f9a-ba3a-ff5999dd444a",
            "x-posthog-window-id": "0199f71f-8c42-7f9a-ba3a-ff5ab5b20a8e",
        }

    def _prepare_payload(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        准备发送给 Smithery.ai 的请求负载

        Args:
            model: 模型名称
            messages: 已转换的消息列表
            tools: 工具定义列表 (OpenAI 格式)
        """
        payload = {
            "messages": messages,
            "tools": self._convert_tools_to_smithery_format(tools or []),
            "model": model,
            "systemPrompt": "You are a helpful assistant."
        }
        return payload

    def _convert_tools_to_smithery_format(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 OpenAI 格式的工具定义转换为 Smithery.ai 格式

        OpenAI 格式:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Smithery 格式:
        {
            "name": "get_weather",
            "description": "Get weather information",
            "inputSchema": {
                "type": "object",
                "properties": {...},
                "required": [...],
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
        """
        smithery_tools = []

        for tool in openai_tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                function = tool.get("function", {})
                name = function.get("name", "")
                description = function.get("description", "")
                parameters = function.get("parameters", {})

                if name:
                    smithery_tool = {
                        "name": name,
                        "description": description,
                        "inputSchema": {
                            **parameters,
                            "$schema": "http://json-schema.org/draft-07/schema#"
                        }
                    }
                    smithery_tools.append(smithery_tool)

        return smithery_tools

    def _format_tool_output(self, tool_output: Dict[str, Any]) -> str:
        """
        格式化工具输出为可读文本
        """
        if isinstance(tool_output, dict):
            content = tool_output.get("content", [])
            if isinstance(content, list) and content:
                first_content = content[0]
                if isinstance(first_content, dict):
                    return first_content.get("text", str(tool_output))
            return str(tool_output)
        return str(tool_output)

    async def get_models(self) -> JSONResponse:
        model_data = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "lzA6"}
                for name in settings.KNOWN_MODELS
            ]
        }
        return JSONResponse(content=model_data)
