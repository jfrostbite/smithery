from cachetools import TTLCache
from typing import List, Dict, Any
from app.core.config import settings

class SessionManager:
    def __init__(self):
        self.cache = TTLCache(maxsize=1024, ttl=settings.SESSION_CACHE_TTL)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        从缓存中获取消息历史。
        返回列表的副本以防止对缓存的意外修改。
        """
        return self.cache.get(session_id, []).copy()

    def update_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """
        将更新后的消息历史存回缓存。
        """
        self.cache[session_id] = messages
