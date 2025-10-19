import os
import json
import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict

# 获取一个日志记录器实例
logger = logging.getLogger(__name__)

class AuthCookie:
    """
    处理并生成 Smithery.ai 所需的认证 Cookie。
    它将 .env 文件中的 JSON 字符串转换为一个标准的 HTTP Cookie 头部字符串。
    """
    def __init__(self, json_string: str):
        try:
            # 1. 解析从 .env 文件读取的 JSON 字符串
            data = json.loads(json_string)
            self.access_token = data.get("access_token")
            self.refresh_token = data.get("refresh_token")
            self.expires_at = data.get("expires_at", 0)
            
            if not self.access_token:
                raise ValueError("Cookie JSON 中缺少 'access_token'")

            # 2. 构造将要放入 Cookie header 的值部分 (它本身也是一个 JSON)
            #    注意：这里我们只包含 Supabase auth 需要的核心字段
            cookie_value_data = {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": data.get("token_type", "bearer"),
                "expires_in": data.get("expires_in", 3600),
                "expires_at": self.expires_at,
                "user": data.get("user")
            }
            
            # 3. 构造完整的 Cookie 键值对字符串
            #    Smithery.ai 使用的 Supabase project_ref 是 'spjawbfpwezjfmicopsl'
            project_ref = "spjawbfpwezjfmicopsl"
            cookie_key = f"sb-{project_ref}-auth-token"
            # 将值部分转换为紧凑的 JSON 字符串
            cookie_value = json.dumps(cookie_value_data, separators=(',', ':'))
            
            # 最终用于 HTTP Header 的字符串，格式为 "key=value"
            self.header_cookie_string = f"{cookie_key}={cookie_value}"

        except json.JSONDecodeError as e:
            raise ValueError(f"无法从提供的字符串中解析认证 JSON: {e}")
        except Exception as e:
            raise ValueError(f"初始化 AuthCookie 时出错: {e}")

    def __repr__(self):
        return f"<AuthCookie expires_at={self.expires_at}>"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra="ignore"
    )

    APP_NAME: str = "smithery-2api"
    APP_VERSION: str = "1.0.0"
    DESCRIPTION: str = "一个将 smithery.ai 转换为兼容 OpenAI 格式 API 的高性能代理，支持多账号、上下文和工具调用。"

    CHAT_API_URL: str = "https://smithery.ai/api/chat"
    TOKEN_REFRESH_URL: str = "https://spjawbfpwezjfmicopsl.supabase.co/auth/v1/token?grant_type=refresh_token"
    SUPABASE_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNwamF3YmZwd2V6amZtaWNvcHNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQxNDc0MDUsImV4cCI6MjA0OTcyMzQwNX0.EBIg7_F2FZh4KZ3UNwZdBRjpp2fgHqXGJOvOSQ053MU"

    API_MASTER_KEY: Optional[str] = None
    
    AUTH_COOKIES: List[AuthCookie] = []

    API_REQUEST_TIMEOUT: int = 180
    APP_PORT: int = 8000
    SESSION_CACHE_TTL: int = 3600

    KNOWN_MODELS: List[str] = [
        "claude-haiku-4.5", "claude-sonnet-4.5", "gpt-5", "gpt-5-mini", 
        "gpt-5-nano", "gemini-2.5-flash-lite", "gemini-2.5-pro", "glm-4.6", 
        "grok-4-fast-non-reasoning", "grok-4-fast-reasoning", "kimi-k2", "deepseek-reasoner"
    ]

    def __init__(self, **values):
        super().__init__(**values)
        # 从环境变量 SMITHERY_COOKIE_1, SMITHERY_COOKIE_2, ... 加载 cookies
        i = 1
        while True:
            cookie_str = os.getenv(f"SMITHERY_COOKIE_{i}")
            if cookie_str:
                try:
                    # 使用 AuthCookie 类来解析和处理 cookie 字符串
                    self.AUTH_COOKIES.append(AuthCookie(cookie_str))
                except ValueError as e:
                    logger.warning(f"无法加载或解析 SMITHERY_COOKIE_{i}: {e}")
                i += 1
            else:
                break

        if not self.AUTH_COOKIES:
            if self.API_MASTER_KEY == "1":  # Development mode
                logger.warning("运行在开发模式：未配置有效的 SMITHERY_COOKIE，API 调用将失败")
                # Create a dummy cookie for development
                self.AUTH_COOKIES = [type('DummyCookie', (), {'header_cookie_string': 'dummy-cookie-for-dev'})()]
            else:
                raise ValueError("必须在 .env 文件中至少配置一个有效的 SMITHERY_COOKIE_1")

settings = Settings()
