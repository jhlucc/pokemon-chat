"""
Agentic Memory - Long-term User Preference Storage

Enables the Agent to remember user preferences across sessions.
Uses SQLite for persistence and LLM for preference extraction.
"""
import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.core.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserPreferences(BaseModel):
    """Extracted user preferences."""
    favorite_pokemon: List[str] = Field(default_factory=list, description="User's favorite Pokemon")
    favorite_types: List[str] = Field(default_factory=list, description="Preferred Pokemon types")
    response_style: str = Field(default="balanced", description="Preferred response style: brief/balanced/detailed")
    interests: List[str] = Field(default_factory=list, description="Topics the user is interested in")
    notes: str = Field(default="", description="Any other notable preferences or context")


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是宝可梦训练师偏好分析专家。
分析对话内容，提取用户的宝可梦偏好：

提取维度：
1. 喜爱的宝可梦 - 用户明确表示喜欢或经常询问的宝可梦
2. 偏好的属性 - 如火系、水系、龙系等
3. 回答风格 - 用户是喜欢简洁明了还是详细深入的回答
4. 兴趣话题 - 对战策略、动画剧情、图鉴收集、竞技培养等
5. 其他备注 - 如游戏版本偏好、最喜欢的世代等

提取原则：
- 保守提取，只记录明确表达或强烈暗示的偏好
- 如果没有明确偏好，保持字段为空
- 注意区分临时询问和真实偏好（反复问同一只宝可梦才算偏好）"""),
    ("human", """对话记录:
{conversation}

已知用户偏好:
{current_preferences}

请提取本次对话中的新偏好（与现有偏好合并）。""")
])


class AgenticMemory:
    """
    Long-term memory for user preferences.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.paths.data_dir / "agentic_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        
        self.llm = ChatOpenAI(
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_base,
            temperature=0
        )
        self.extraction_chain = EXTRACTION_PROMPT | self.llm.with_structured_output(UserPreferences)
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT preferences FROM user_preferences WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                data = json.loads(row[0])
                return UserPreferences(**data)
            except Exception as e:
                logger.warning(f"Failed to parse preferences: {e}")
        
        return UserPreferences()
    
    def save_preferences(self, user_id: str, preferences: UserPreferences):
        """Save user preferences."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences (user_id, preferences, updated_at)
            VALUES (?, ?, ?)
        """, (user_id, preferences.model_dump_json(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.debug(f"Saved preferences for user {user_id}")
    
    def add_conversation_turn(self, user_id: str, role: str, content: str):
        """Add a conversation turn to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_history (user_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, role, content, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_recent_conversation(self, user_id: str, limit: int = 10) -> str:
        """Get recent conversation as formatted string."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content FROM conversation_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        rows = rows[::-1]
        
        return "\n".join([f"{role}: {content}" for role, content in rows])
    
    def extract_and_update_preferences(self, user_id: str) -> UserPreferences:
        """
        Extract preferences from recent conversation and update stored preferences.
        Should be called periodically (e.g., every N turns).
        """
        current_prefs = self.get_preferences(user_id)
        conversation = self.get_recent_conversation(user_id, limit=20)
        
        if not conversation:
            return current_prefs
        
        try:
            new_prefs = self.extraction_chain.invoke({
                "conversation": conversation,
                "current_preferences": current_prefs.model_dump_json(indent=2)
            })
            
            # Merge preferences (new takes precedence for non-list fields)
            merged = UserPreferences(
                favorite_pokemon=list(set(current_prefs.favorite_pokemon + new_prefs.favorite_pokemon)),
                favorite_types=list(set(current_prefs.favorite_types + new_prefs.favorite_types)),
                response_style=new_prefs.response_style if new_prefs.response_style != "balanced" else current_prefs.response_style,
                interests=list(set(current_prefs.interests + new_prefs.interests)),
                notes=new_prefs.notes or current_prefs.notes
            )
            
            self.save_preferences(user_id, merged)
            logger.info(f"Updated preferences for user {user_id}")
            return merged
            
        except Exception as e:
            logger.error(f"Failed to extract preferences: {e}")
            return current_prefs
    
    def get_system_prompt_injection(self, user_id: str) -> str:
        """
        Generate a system prompt injection based on user preferences.
        """
        prefs = self.get_preferences(user_id)
        
        parts = []
        
        if prefs.favorite_pokemon:
            parts.append(f"User's favorite Pokemon: {', '.join(prefs.favorite_pokemon)}")
        
        if prefs.favorite_types:
            parts.append(f"User prefers {', '.join(prefs.favorite_types)} type Pokemon")
        
        if prefs.response_style == "brief":
            parts.append("User prefers brief, concise answers")
        elif prefs.response_style == "detailed":
            parts.append("User prefers detailed, comprehensive answers")
        
        if prefs.interests:
            parts.append(f"User is interested in: {', '.join(prefs.interests)}")
        
        if prefs.notes:
            parts.append(f"Additional context: {prefs.notes}")
        
        if not parts:
            return ""
        
        return "\n\n[User Preferences]\n" + "\n".join(f"- {p}" for p in parts)


# Global instance
_memory: AgenticMemory = None

def get_agentic_memory() -> AgenticMemory:
    global _memory
    if _memory is None:
        _memory = AgenticMemory()
    return _memory
