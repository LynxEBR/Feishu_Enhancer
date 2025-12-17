"""
数据库模型定义
"""
from sqlalchemy import Column, BigInteger, Text, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # 如果 pgvector 未安装，使用字符串类型作为后备
    from sqlalchemy import Text as Vector
from reasoning_knowledge.database import Base


class AIReasoningKnowledge(Base):
    """AI 推理知识库表模型"""
    __tablename__ = "ai_reasoning_knowledge"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    task_text = Column(Text, nullable=False, comment="任务文本")
    step_text = Column(Text, nullable=False, comment="步骤文本")
    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间"
    )
    task_embedding = Column(
        Vector(1024),
        nullable=False,
        comment="任务向量嵌入"
    )
    step_embedding = Column(
        Vector(1024),
        nullable=False,
        comment="步骤向量嵌入"
    )

    def __repr__(self):
        return f"<AIReasoningKnowledge(id={self.id}, task='{self.task_text[:50]}...')>"

    def to_dict(self):
        """转换为字典格式（不包含向量）"""
        return {
            "id": self.id,
            "task_text": self.task_text,
            "step_text": self.step_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

