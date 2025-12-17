"""
任务存储数据库模型定义
"""
import json
from sqlalchemy import Column, BigInteger, Text, Boolean, func
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSON
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # 如果 pgvector 未安装，使用字符串类型作为后备
    from sqlalchemy import Text as Vector
from task_storage.database import Base


class AITask(Base):
    """AI 任务表模型"""
    __tablename__ = "ai_task"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    original_task = Column(Text, nullable=True, comment="原始任务")
    enhanced_task = Column(Text, nullable=True, comment="增强后的任务")
    can_execute = Column(Boolean, nullable=True, comment="是否可执行")
    execution_reason = Column(Text, nullable=True, comment="执行原因")
    steps = Column(Text, nullable=True, comment="步骤")
    step_results = Column(JSON, nullable=True, comment="步骤结果")
    final_result = Column(Text, nullable=True, comment="最终结果")
    enhanced_task_embedding = Column(
        Vector(1024),
        nullable=True,
        comment="增强任务的向量嵌入"
    )
    all_success = Column(Boolean, nullable=True, comment="是否全部成功")

    def __repr__(self):
        return f"<AITask(id={self.id}, original_task='{self.original_task[:50] if self.original_task else None}...')>"

    def to_dict(self):
        """转换为字典格式（不包含向量）"""
        return {
            "id": self.id,
            "original_task": self.original_task,
            "enhanced_task": self.enhanced_task,
            "can_execute": self.can_execute,
            "execution_reason": self.execution_reason,
            "steps": self.steps,
            "step_results": self.step_results,
            "final_result": self.final_result,
            "all_success": self.all_success,
        }

