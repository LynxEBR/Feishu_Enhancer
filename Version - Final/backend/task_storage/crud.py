"""
任务存储 CRUD 操作服务
"""
from typing import List, Optional, Dict, Any
import json
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from task_storage.models import AITask
from business_knowledge.embedding_service import get_embedding_service


class TaskStorageCRUD:
    """任务存储 CRUD 操作类"""
    
    def __init__(self, db: Session):
        """
        初始化 CRUD 服务
        
        Args:
            db: 数据库会话
        """
        self.db = db
        self.embedding_service = get_embedding_service()
    
    def create(
        self,
        original_task: Optional[str] = None,
        enhanced_task: Optional[str] = None,
        can_execute: Optional[bool] = None,
        execution_reason: Optional[str] = None,
        steps: Optional[str] = None,
        step_results: Optional[Dict[str, Any]] = None,
        final_result: Optional[str] = None,
        all_success: Optional[bool] = None
    ) -> AITask:
        """
        创建新的任务记录
        
        Args:
            original_task: 原始任务
            enhanced_task: 增强后的任务（如果不为 None，会自动生成 embedding）
            can_execute: 是否可执行
            execution_reason: 执行原因
            steps: 步骤（字符串格式）
            step_results: 步骤结果（字典格式）
            final_result: 最终结果
            all_success: 是否全部成功
            
        Returns:
            创建的任务对象
        """
        # 如果 enhanced_task 不为 None，生成 embedding
        enhanced_task_embedding = None
        if enhanced_task is not None:
            embedding = self.embedding_service.encode_question(enhanced_task)
            # 确保是单个向量（不是批量）
            if embedding.ndim > 1:
                embedding = embedding[0]
            # 转换为列表格式（pgvector 需要）
            enhanced_task_embedding = embedding.tolist()
        
        task = AITask(
            original_task=original_task,
            enhanced_task=enhanced_task,
            can_execute=can_execute,
            execution_reason=execution_reason,
            steps=steps,
            step_results=step_results,
            final_result=final_result,
            enhanced_task_embedding=enhanced_task_embedding,
            all_success=all_success
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def update(
        self,
        task_id: int,
        original_task: Optional[str] = None,
        enhanced_task: Optional[str] = None,
        can_execute: Optional[bool] = None,
        execution_reason: Optional[str] = None,
        steps: Optional[str] = None,
        step_results: Optional[Dict[str, Any]] = None,
        final_result: Optional[str] = None,
        all_success: Optional[bool] = None
    ) -> Optional[AITask]:
        """
        更新任务记录
        
        Args:
            task_id: 任务 ID
            original_task: 原始任务
            enhanced_task: 增强后的任务（如果不为 None，会自动生成 embedding）
            can_execute: 是否可执行
            execution_reason: 执行原因
            steps: 步骤（字符串格式）
            step_results: 步骤结果（字典格式）
            final_result: 最终结果
            all_success: 是否全部成功
            
        Returns:
            更新后的任务对象，如果不存在则返回 None
        """
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        if original_task is not None:
            task.original_task = original_task
        if enhanced_task is not None:
            task.enhanced_task = enhanced_task
            # 如果 enhanced_task 不为 None，生成新的 embedding
            embedding = self.embedding_service.encode_question(enhanced_task)
            # 确保是单个向量（不是批量）
            if embedding.ndim > 1:
                embedding = embedding[0]
            # 转换为列表格式（pgvector 需要）
            task.enhanced_task_embedding = embedding.tolist()
        if can_execute is not None:
            task.can_execute = can_execute
        if execution_reason is not None:
            task.execution_reason = execution_reason
        if steps is not None:
            task.steps = steps
        if step_results is not None:
            task.step_results = step_results
        if final_result is not None:
            task.final_result = final_result
        if all_success is not None:
            task.all_success = all_success
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def get_by_id(self, task_id: int) -> Optional[AITask]:
        """
        根据 ID 获取任务记录
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务对象，如果不存在则返回 None
        """
        return self.db.query(AITask).filter(AITask.id == task_id).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[AITask]:
        """
        获取所有任务记录（分页）
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            任务列表
        """
        return self.db.query(AITask).offset(skip).limit(limit).all()
    
    def delete(self, task_id: int) -> bool:
        """
        删除任务记录
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否删除成功
        """
        task = self.get_by_id(task_id)
        if not task:
            return False
        
        self.db.delete(task)
        self.db.commit()
        
        return True
    
    def search_by_enhanced_task(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        根据增强任务文本进行向量相似度搜索
        
        Args:
            query_text: 查询任务文本
            top_k: 返回最相似的前 k 个结果
            threshold: 相似度阈值（0-1），低于此值的结果将被过滤
            
        Returns:
            搜索结果列表，每个结果包含任务信息和相似度分数
        """
        # 生成查询向量
        query_embedding = self.embedding_service.encode_question(query_text)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]
        query_embedding_list = query_embedding.tolist()
        
        # 使用余弦相似度进行搜索
        # pgvector 使用 <=> 操作符计算余弦距离（1 - 余弦相似度）
        # 所以距离越小，相似度越高
        try:
            from pgvector.sqlalchemy import Vector
            # 使用原生 SQL 进行向量相似度搜索
            # 注意：使用 CAST 代替 :: 语法，避免与 SQLAlchemy 参数绑定冲突
            query_vector_str = '[' + ','.join(map(str, query_embedding_list)) + ']'
            results = self.db.execute(
                text("""
                    SELECT 
                        id, original_task, enhanced_task, can_execute, 
                        execution_reason, steps, step_results, final_result, all_success,
                        1 - (enhanced_task_embedding <=> CAST(:query_vector AS vector)) as similarity
                    FROM ai_task
                    WHERE enhanced_task_embedding IS NOT NULL
                    ORDER BY enhanced_task_embedding <=> CAST(:query_vector AS vector)
                    LIMIT :top_k
                """),
                {
                    "query_vector": query_vector_str,
                    "top_k": top_k
                }
            ).fetchall()
            
            # 转换为任务对象
            search_results = []
            for row in results:
                if row.similarity >= threshold:
                    task = self.get_by_id(row.id)
                    if task:
                        result_dict = task.to_dict()
                        result_dict['similarity'] = float(row.similarity)
                        search_results.append(result_dict)
            return search_results
        except Exception as e:
            # 如果 pgvector 不可用，使用简单的文本匹配
            # 回滚事务，避免后续查询失败
            self.db.rollback()
            print(f"向量搜索失败，使用文本匹配: {str(e)}")
            results = self.db.query(AITask).filter(
                AITask.enhanced_task.isnot(None),
                AITask.enhanced_task.ilike(f"%{query_text}%")
            ).limit(top_k).all()
            
            search_results = []
            for task in results:
                result_dict = task.to_dict()
                result_dict['similarity'] = 0.5  # 默认相似度
                search_results.append(result_dict)
            return search_results
    
    def count(self) -> int:
        """
        获取任务存储中的总记录数
        
        Returns:
            总记录数
        """
        return self.db.query(AITask).count()

