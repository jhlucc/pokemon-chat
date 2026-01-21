import os
import pathlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.settings import settings

class DBManager:
    """数据库管理器"""

    def __init__(self):
        # Keep server sqlite files under the same save root as the rest of the app.
        self.db_path = os.path.join(str(settings.paths.save_yaml_path), "data", "server.db")
        self.ensure_db_dir()

        # 创建SQLAlchemy引擎
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        # 创建会话工厂
        self.Session = sessionmaker(bind=self.engine)

        # 确保表存在
        self.create_tables()

    def ensure_db_dir(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        pathlib.Path(db_dir).mkdir(parents=True, exist_ok=True)

    def create_tables(self):
        """创建数据库表"""
        # Lazy import: only needed when the admin token DB is actually used.
        from src.models.token_model import Base

        Base.metadata.create_all(self.engine)

    def get_session(self):
        """获取数据库会话"""
        return self.Session()

_db_manager_instance: DBManager | None = None


def get_db_manager() -> DBManager:
    """Lazy DBManager singleton to keep server startup cheap."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DBManager()
    return _db_manager_instance


class _DBManagerProxy:
    """Backward-compatible lazy proxy for the old `db_manager` global."""

    def __getattr__(self, name):
        return getattr(get_db_manager(), name)

    def __repr__(self) -> str:  # pragma: no cover
        return "<DBManagerProxy lazy>"


# Backward-compatible alias (lazy).
db_manager = _DBManagerProxy()
