<div align="center">

# 🤖 smithery-2api 🤖

**将 [Smithery.ai](https://smithery.ai/) 强大的 AI 模型能力无缝转换为兼容 OpenAI API 格式的转换器**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Docker Support](https://img.shields.io/badge/docker-supported-blue.svg?logo=docker)](https://www.docker.com/)
[![GitHub Repo stars](https://img.shields.io/github/stars/lzA6/smithery-2api?style=social)](https://github.com/lzA6/smithery-2api)

</div>

> "任何足够先进的技术，都与魔法无异。" —— 亚瑟·克拉克
>
> 我们不创造魔法，我们只是让每个人都能成为魔法师。`smithery-2api` 的诞生，源于一个简单的信念：强大的工具应该被更广泛、更便捷地使用。我们希望通过这个项目，打破平台的壁垒，将 Smithery.ai 先进的模型能力，注入到每一个支持 OpenAI API 的应用生态中。

---

## ✨ 核心特性

*   **🚀 零成本接入** - 免费将 Smithery.ai 的多种模型接入现有 OpenAI 生态
*   **🔌 高度兼容** - 完全模拟 OpenAI 的 `/v1/chat/completions` 和 `/v1/models` 接口
*   **🛠️ 工具调用支持** - 完整支持 OpenAI Function Calling (Tools) 功能，实时流式工具执行
*   **🖼️ 多模态支持** - 支持文本和图像输入，兼容 OpenAI 的多模态 API 格式
*   **💨 实时流式响应** - 保持原生 SSE 流式体验，支持文本和工具调用的增量传输
*   **🔄 多账号轮询** - 支持配置多个 Smithery.ai 账号，自动轮询提高稳定性
*   **💨 无状态设计** - 极致轻量，易于水平扩展，保护用户隐私
*   **☁️ 穿透 Cloudflare** - 内置自动处理 Cloudflare 防护机制
*   **📦 Docker 一键部署** - 一条命令即可启动服务
*   **🔓 开源自由** - 采用 Apache 2.0 协议，自由使用、修改和分发

---

## 🏗️ 架构设计

### 核心工作原理

`smithery-2api` 充当一个智能的协议转换器，在 OpenAI API 格式和 Smithery.ai 内部 API 格式之间进行实时转换。

```mermaid
graph TB
    Client[第三方应用] -->|OpenAI 格式请求| API[smithery-2api]
    API -->|协议转换| Translator[API 转换器]
    Translator -->|身份伪装| Identity[身份伪装模块]
    Identity -->|Smithery 格式请求| Smithery[Smithery.ai 服务]
    Smithery -->|流式响应| API
    API -->|OpenAI 格式响应| Client
    
    subgraph "转换过程详解"
        Translator --> Format[格式转换]
        Identity --> Headers[请求头模拟]
        Identity --> Cookie[Cookie 处理]
        Identity --> Cloudflare[Cloudflare 穿透]
    end
```

### 技术实现细节

#### 1. API 格式转换

**技术核心**: `app/providers/smithery_provider.py` 中的 `_convert_messages_to_smithery_format` 方法

**转换示例**:

**输入 (OpenAI 格式)**:
```json
{
  "model": "claude-haiku-4.5",
  "messages": [
    {"role": "user", "content": "你好，请介绍一下自己"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "城市名"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": true
}
```

**输出 (Smithery.ai 格式)**:
```json
{
  "model": "claude-haiku-4.5",
  "messages": [
    {
      "role": "user",
      "parts": [{"type": "text", "text": "你好，请介绍一下自己"}],
      "id": "msg-xxxxxxxx"
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "获取天气信息",
      "inputSchema": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "城市名"}
        },
        "required": ["location"],
        "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  ],
  "systemPrompt": "You are a helpful assistant."
}
```

#### 2. 身份认证与伪装

**技术核心**: `app/core/config.py` 中的 `AuthCookie` 类

```python
class AuthCookie(BaseModel):
    """Smithery 认证 Cookie 数据结构"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str
    user: Dict[str, Any]
    
    @property
    def header_cookie_string(self) -> str:
        """生成请求头中的 Cookie 字符串"""
        return f"sb-access-token={self.access_token}; sb-refresh-token={self.refresh_token}"
```

#### 3. 流式响应处理

**技术核心**: `app/utils/sse_utils.py` 和 `app/providers/smithery_provider.py` 中的流式处理逻辑

```python
async def stream_generator() -> AsyncGenerator[bytes, None]:
    """处理 Smithery.ai 流式响应并转换为 OpenAI 格式"""

    # 维护工具调用状态
    current_tool_call = {"id": None, "name": None}

    for line in response.iter_lines():
        if line.startswith(b"data:"):
            data = json.loads(line[6:])
            response_type = data.get("type", "")

            # 处理文本流式响应
            if response_type == "text-delta":
                delta_content = data.get("delta", "")
                chunk = create_chat_completion_chunk(request_id, model, delta_content)
                yield create_sse_data(chunk)

            # 处理工具调用开始
            elif response_type == "tool-input-start":
                tool_call_id = data.get("toolCallId")
                tool_name = data.get("toolName")
                current_tool_call.update({"id": tool_call_id, "name": tool_name})
                chunk = create_tool_call_chunk(request_id, model, tool_call_id, tool_name)
                yield create_sse_data(chunk)

            # 处理工具参数增量数据
            elif response_type == "tool-input-delta":
                input_text_delta = data.get("inputTextDelta", "")
                if input_text_delta:
                    chunk = create_tool_call_chunk(
                        request_id, model,
                        current_tool_call["id"],
                        current_tool_call["name"],
                        input_text_delta
                    )
                    yield create_sse_data(chunk)
```

#### 4. 多模态内容处理

**技术核心**: 支持文本和图像的混合输入

```python
def _convert_messages_to_smithery_format(self, openai_messages):
    """支持多模态内容转换"""
    for msg in openai_messages:
        content = msg.get("content")
        parts = []

        if isinstance(content, list):  # 多模态内容
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image/"):
                        media_type = self._extract_media_type_from_data_url(image_url)
                        parts.append({
                            "type": "file",
                            "mediaType": media_type,
                            "filename": f"image.{media_type.split('/')[1]}",
                            "url": image_url
                        })
        elif isinstance(content, str):  # 纯文本内容
            parts.append({"type": "text", "text": content})
```

---

## 🚀 快速开始

### 环境要求

- **Git** - 版本控制工具
- **Docker & Docker Compose** - 容器化部署

### 部署步骤

#### 步骤 1: 获取项目代码

```bash
git clone https://github.com/lzA6/smithery-2api.git
cd smithery-2api
```

#### 步骤 2: 获取认证信息

1. 在浏览器中登录 [Smithery.ai](https://smithery.ai/)
2. 打开开发者工具 (F12)
3. 切换到 **Application** → **Local Storage** → `https://smithery.ai`
4. 找到 key 为 `sb-spjawbfpwezjfmicopsl-auth-token` 的项
5. 复制完整的 value 值（JSON 格式）

![获取 Cookie 演示](https://i.imgur.com/aY2i1sA.gif)

#### 步骤 3: 配置环境变量

```bash
# 复制环境配置模板
cp .env.example .env

# 编辑配置文件
vim .env
```

**环境变量配置示例**:
```env
# API 主密钥（用于客户端认证）
API_MASTER_KEY="your-secure-master-key-here"

# 服务端口配置（默认 8000，可自定义）
APP_PORT=3004

# Smithery.ai 认证信息（支持多个账号轮询）
SMITHERY_COOKIE_1='{"access_token":"eyJ...","token_type":"bearer","expires_in":3600,...}'
SMITHERY_COOKIE_2='{"access_token":"eyJ...","token_type":"bearer","expires_in":3600,...}'

# 会话管理（可选）
SESSION_CACHE_TTL=3600
```

#### 步骤 4: 启动服务

**Docker 方式（推荐）**:
```bash
docker-compose up -d
```

**本地开发方式**:
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn main:app --reload --port 8000
```

#### 步骤 5: 验证部署

使用 curl 测试服务是否正常运行：

```bash
# 测试模型列表接口
curl -X GET "http://localhost:3004/v1/models" \
  -H "Authorization: Bearer your-secure-master-key-here"

# 测试聊天补全接口
curl -X POST "http://localhost:3004/v1/chat/completions" \
  -H "Authorization: Bearer your-secure-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# 测试工具调用功能
curl -X POST "http://localhost:3004/v1/chat/completions" \
  -H "Authorization: Bearer your-secure-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4.5",
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "stream": true
  }'
```

### 客户端配置示例

**OpenAI 官方客户端**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3004/v1",
    api_key="your-secure-master-key-here"
)

# 文本聊天
response = client.chat.completions.create(
    model="claude-haiku-4.5",
    messages=[{"role": "user", "content": "Hello, world!"}],
    stream=True
)

# 工具调用示例
response = client.chat.completions.create(
    model="claude-haiku-4.5",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    stream=True
)

# 多模态输入示例
response = client.chat.completions.create(
    model="claude-haiku-4.5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."}
                }
            ]
        }
    ],
    stream=True
)
```

**第三方应用配置**:
- **Base URL**: `http://localhost:3004/v1`
- **API Key**: `your-secure-master-key-here`
- **支持的功能**:
  - ✅ 文本聊天
  - ✅ 流式响应
  - ✅ 工具调用 (Function Calling)
  - ✅ 多模态输入 (文本+图像)
  - ✅ 所有 Smithery.ai 支持的模型

---

## 🛠️ 技术架构

### 技术栈

| 技术组件 | 版本 | 用途 | 关键特性 |
|---------|------|------|----------|
| **FastAPI** | 0.104+ | Web 框架 | 异步支持，自动文档生成 |
| **Pydantic** | 2.5+ | 数据验证 | 类型提示，配置管理 |
| **Cloudscraper** | 1.2+ | 反爬虫绕过 | Cloudflare 穿透 |
| **Uvicorn** | 0.24+ | ASGI 服务器 | 高性能异步服务器 |
| **Docker** | 20.10+ | 容器化 | 环境隔离，一键部署 |
| **Nginx** | 1.24+ | 反向代理 | 负载均衡，静态文件服务 |

### 项目结构

```
smithery-2api/
├── 📁 app/                          # 应用核心代码
│   ├── 📁 core/                     # 核心模块
│   │   ├── __init__.py
│   │   └── config.py               # 配置管理，环境变量处理
│   ├── 📁 providers/               # 服务提供商模块
│   │   ├── __init__.py
│   │   ├── base_provider.py        # 提供商基类
│   │   └── smithery_provider.py    # Smithery.ai 提供商实现
│   ├── 📁 services/                # 业务服务层
│   │   ├── session_manager.py      # 会话管理（预留）
│   │   └── tool_caller.py          # 工具调用（预留）
│   └── 📁 utils/                   # 工具函数
│       └── sse_utils.py           # Server-Sent Events 工具
├── 📄 main.py                      # FastAPI 应用入口
├── 📄 nginx.conf                   # Nginx 配置
├── 📄 Dockerfile                   # 应用镜像构建配置
├── 📄 docker-compose.yml           # 服务编排配置
├── 📄 requirements.txt             # Python 依赖
└── 📄 .env.example                 # 环境变量模板
```

### 核心模块详解

#### 1. 配置管理 (`app/core/config.py`)

```python
class Settings(BaseSettings):
    """应用配置类"""
    API_MASTER_KEY: str
    SMITHERY_COOKIE_1: Optional[str] = None
    SMITHERY_COOKIE_2: Optional[str] = None
    NGINX_PORT: int = 8088
    APP_PORT: int = 8000
    
    @property
    def AUTH_COOKIES(self) -> List[AuthCookie]:
        """获取所有可用的认证 Cookie"""
        cookies = []
        for i in range(1, 3):
            if cookie_str := getattr(self, f"SMITHERY_COOKIE_{i}", None):
                try:
                    cookies.append(AuthCookie.parse_raw(cookie_str))
                except ValidationError as e:
                    logger.warning(f"Invalid cookie format for SMITHERY_COOKIE_{i}: {e}")
        return cookies

settings = Settings()
```

#### 2. Smithery 提供商 (`app/providers/smithery_provider.py`)

```python
class SmitheryProvider(BaseProvider):
    """Smithery.ai 服务提供商"""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> StreamingResponse:
        """处理聊天补全请求"""
        
        # 1. 轮询获取可用 Cookie
        auth_cookie = self._get_cookie()
        
        # 2. 准备请求头和负载
        headers = self._prepare_headers(auth_cookie)
        payload = self._prepare_payload(request)
        
        # 3. 发送请求到 Smithery.ai
        response = self._make_request(headers, payload)
        
        # 4. 处理流式响应
        return self._handle_stream_response(response, request.model)

    def _convert_messages_to_smithery_format(self, messages: List[Dict]) -> List[Dict]:
        """将 OpenAI 消息格式转换为 Smithery 格式"""
        converted_messages = []
        for msg in messages:
            converted_msg = {
                "role": msg["role"],
                "parts": [{"type": "text", "text": msg["content"]}],
                "id": f"msg-{str(uuid.uuid4())[:8]}"
            }
            converted_messages.append(converted_msg)
        return converted_messages
```

---

## 🔧 高级配置

### 多账号负载均衡

支持配置多个 Smithery.ai 账号实现自动轮询：

```env
# 配置多个账号提高可用性
SMITHERY_COOKIE_1='{"access_token":"token1","refresh_token":"refresh1",...}'
SMITHERY_COOKIE_2='{"access_token":"token2","refresh_token":"refresh2",...}'
SMITHERY_COOKIE_3='{"access_token":"token3","refresh_token":"refresh3",...}'
```

### 自定义模型映射

在 `smithery_provider.py` 中配置模型映射关系：

```python
MODEL_MAPPING = {
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "claude-haiku-4.5", 
    "claude-3-opus": "claude-opus-3.0"
}
```

### Nginx 优化配置

```nginx
# nginx.conf
server {
    listen 8088;
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 流式响应相关配置
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
```

---

## 🐛 故障排除

### 常见问题

**1. 认证失败**
```
错误信息: Authentication failed or cookie expired
解决方案: 重新获取 SMITHERY_COOKIE 值
```

**2. Cloudflare 拦截**
```
错误信息: 403 Forbidden or Cloudflare challenge
解决方案: 确保 cloudscraper 版本最新，或更新请求头信息
```

**3. 流式响应中断**
```
错误信息: 连接提前关闭
解决方案: 检查客户端超时设置，确保网络稳定性
```

### 日志调试

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 健康检查

```bash
# 检查服务状态
curl -X GET "http://localhost:8088/health"

# 检查模型列表
curl -X GET "http://localhost:8088/v1/models" \
  -H "Authorization: Bearer your-api-key"
```

---

## 🚧 限制与待完善功能

### 当前限制

1. **令牌过期处理** - 当前未实现自动刷新机制
2. **错误处理** - 错误信息返回可以更加友好
3. **速率限制** - 缺少请求频率限制
4. **会话管理** - 无状态设计，客户端需维护完整上下文

### 开发路线图

- [ ] **自动令牌刷新机制**
  - 利用 refresh_token 自动更新 access_token
  - 实现令牌过期前预刷新

- [ ] **增强的错误处理**
  - 更友好的错误消息
  - 重试机制和故障转移

- [ ] **会话状态管理**
  - 可选的有状态会话模式
  - Redis 后端支持

- [ ] **监控和指标**
  - Prometheus 指标收集
  - 请求统计和性能监控

- [ ] **扩展提供商支持**
  - 支持其他 AI 服务平台
  - 统一的提供商接口

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 报告问题

如果您发现任何问题，请通过 [GitHub Issues](https://github.com/lzA6/smithery-2api/issues) 报告。

### 代码贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/lzA6/smithery-2api.git
cd smithery-2api

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
cp .env.example .env
# 编辑 .env 文件配置认证信息

# 启动开发服务器
uvicorn main:app --reload --port 8000
```

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者，以及 Smithery.ai 提供的优秀 AI 服务。

---

## 🔗 有用链接

- [Smithery.ai 官网](https://smithery.ai/)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Docker 文档](https://docs.docker.com/)

---

<div align="center">

**如果这个项目对您有帮助，请给个 ⭐️ 支持一下！**

</div>
