"""
数据库连接配置
"""
import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# 导入 config 模块
try:
    # 尝试相对导入（当作为包导入时）
    from .. import config
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from backend import config
    except ImportError:
        # 如果都失败，添加项目根目录到路径
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from backend import config

# 从配置文件或环境变量获取数据库连接信息
DATABASE_URL = config.config_dict.get(
    "DATABASE_URL",
    os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/ai_test_db"
    )
)

# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    echo=False,  # 设置为 True 可以查看 SQL 语句
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基础模型类
Base = declarative_base()


def init_db():
    """初始化数据库，创建表结构"""
    # # 首先确保 pgvector 扩展已安装
    # with engine.connect() as conn:
    #     # 检查并创建 pgvector 扩展
    #     conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    #     conn.commit()
    
    # 创建所有表
    Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库会话（用于依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

