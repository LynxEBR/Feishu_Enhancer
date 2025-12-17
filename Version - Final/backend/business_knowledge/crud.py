"""
CRUD 操作服务
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np
from business_knowledge.models import AIBusinessKnowledge
from business_knowledge.embedding_service import get_embedding_service


class BusinessKnowledgeCRUD:
    """业务知识库 CRUD 操作类"""
    
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
        question_text: str,
        answer_text: str
    ) -> AIBusinessKnowledge:
        """
        创建新的知识条目
        
        Args:
            question_text: 问题文本
            answer_text: 答案文本
            
        Returns:
            创建的知识条目对象
        """
        # 生成 embedding
        question_embedding = self.embedding_service.encode_question(question_text)
        answer_embedding = self.embedding_service.encode_answer(answer_text)
        
        # 确保是单个向量（不是批量）
        if question_embedding.ndim > 1:
            question_embedding = question_embedding[0]
        if answer_embedding.ndim > 1:
            answer_embedding = answer_embedding[0]
        
        # 转换为列表格式（pgvector 需要）
        question_embedding_list = question_embedding.tolist()
        answer_embedding_list = answer_embedding.tolist()
        
        # 创建新记录
        knowledge = AIBusinessKnowledge(
            question_text=question_text,
            answer_text=answer_text,
            question_embedding=question_embedding_list,
            answer_embedding=answer_embedding_list
        )
        
        self.db.add(knowledge)
        self.db.commit()
        self.db.refresh(knowledge)
        
        return knowledge
    
    def get_by_id(self, knowledge_id: int) -> Optional[AIBusinessKnowledge]:
        """
        根据 ID 获取知识条目
        
        Args:
            knowledge_id: 知识条目 ID
            
        Returns:
            知识条目对象，如果不存在则返回 None
        """
        return self.db.query(AIBusinessKnowledge).filter(
            AIBusinessKnowledge.id == knowledge_id
        ).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIBusinessKnowledge]:
        """
        获取所有知识条目（分页）
        
        Args:
            skip: 跳过的记录数
            limit: 返回的最大记录数
            
        Returns:
            知识条目列表
        """
        return self.db.query(AIBusinessKnowledge).offset(skip).limit(limit).all()
    
    def update(
        self,
        knowledge_id: int,
        question_text: Optional[str] = None,
        answer_text: Optional[str] = None
    ) -> Optional[AIBusinessKnowledge]:
        """
        更新知识条目
        
        Args:
            knowledge_id: 知识条目 ID
            question_text: 新的问题文本（可选）
            answer_text: 新的答案文本（可选）
            
        Returns:
            更新后的知识条目对象，如果不存在则返回 None
        """
        knowledge = self.get_by_id(knowledge_id)
        if not knowledge:
            return None
        
        # 更新文本
        if question_text is not None:
            knowledge.question_text = question_text
            # 重新生成 embedding
            question_embedding = self.embedding_service.encode_question(question_text)
            if question_embedding.ndim > 1:
                question_embedding = question_embedding[0]
            knowledge.question_embedding = question_embedding.tolist()
        
        if answer_text is not None:
            knowledge.answer_text = answer_text
            # 重新生成 embedding
            answer_embedding = self.embedding_service.encode_answer(answer_text)
            if answer_embedding.ndim > 1:
                answer_embedding = answer_embedding[0]
            knowledge.answer_embedding = answer_embedding.tolist()
        
        self.db.commit()
        self.db.refresh(knowledge)
        
        return knowledge
    
    def delete(self, knowledge_id: int) -> bool:
        """
        删除知识条目
        
        Args:
            knowledge_id: 知识条目 ID
            
        Returns:
            是否删除成功
        """
        knowledge = self.get_by_id(knowledge_id)
        if not knowledge:
            return False
        
        self.db.delete(knowledge)
        self.db.commit()
        
        return True
    
    def search_by_question(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        根据问题文本进行向量相似度搜索
        
        Args:
            query_text: 查询问题文本
            top_k: 返回最相似的前 k 个结果
            threshold: 相似度阈值（0-1），低于此值的结果将被过滤
            
        Returns:
            搜索结果列表，每个结果包含知识条目信息和相似度分数
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
                        id, question_text, answer_text, created_at, updated_at,
                        1 - (question_embedding <=> CAST(:query_vector AS vector)) as similarity
                    FROM ai_business_knowledge
                    ORDER BY question_embedding <=> CAST(:query_vector AS vector)
                    LIMIT :top_k
                """),
                {
                    "query_vector": query_vector_str,
                    "top_k": top_k
                }
            ).fetchall()
            
            # 转换为知识条目对象
            search_results = []
            for row in results:
                if row.similarity >= threshold:
                    knowledge = self.get_by_id(row.id)
                    if knowledge:
                        result_dict = knowledge.to_dict()
                        result_dict['similarity'] = float(row.similarity)
                        search_results.append(result_dict)
            return search_results
        except Exception as e:
            # 如果 pgvector 不可用，使用简单的文本匹配
            # 回滚事务，避免后续查询失败
            self.db.rollback()
            print(f"向量搜索失败，使用文本匹配: {str(e)}")
            results = self.db.query(AIBusinessKnowledge).filter(
                AIBusinessKnowledge.question_text.ilike(f"%{query_text}%")
            ).limit(top_k).all()
            
            search_results = []
            for knowledge in results:
                result_dict = knowledge.to_dict()
                result_dict['similarity'] = 0.5  # 默认相似度
                search_results.append(result_dict)
            return search_results
    
    def search_by_answer(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        根据答案文本进行向量相似度搜索
        
        Args:
            query_text: 查询文本
            top_k: 返回最相似的前 k 个结果
            threshold: 相似度阈值（0-1），低于此值的结果将被过滤
            
        Returns:
            搜索结果列表，每个结果包含知识条目信息和相似度分数
        """
        # 生成查询向量
        query_embedding = self.embedding_service.encode_answer(query_text)
        if query_embedding.ndim > 1:
            query_embedding = query_embedding[0]
        query_embedding_list = query_embedding.tolist()
        
        # 使用余弦相似度进行搜索
        try:
            from pgvector.sqlalchemy import Vector
            # 使用原生 SQL 进行向量相似度搜索
            # 注意：使用 CAST 代替 :: 语法，避免与 SQLAlchemy 参数绑定冲突
            query_vector_str = '[' + ','.join(map(str, query_embedding_list)) + ']'
            results = self.db.execute(
                text("""
                    SELECT 
                        id, question_text, answer_text, created_at, updated_at,
                        1 - (answer_embedding <=> CAST(:query_vector AS vector)) as similarity
                    FROM ai_business_knowledge
                    ORDER BY answer_embedding <=> CAST(:query_vector AS vector)
                    LIMIT :top_k
                """),
                {
                    "query_vector": query_vector_str,
                    "top_k": top_k
                }
            ).fetchall()
            
            # 转换为知识条目对象
            search_results = []
            for row in results:
                if row.similarity >= threshold:
                    knowledge = self.get_by_id(row.id)
                    if knowledge:
                        result_dict = knowledge.to_dict()
                        result_dict['similarity'] = float(row.similarity)
                        search_results.append(result_dict)
            return search_results
        except Exception as e:
            # 如果 pgvector 不可用，使用简单的文本匹配
            # 回滚事务，避免后续查询失败
            self.db.rollback()
            print(f"向量搜索失败，使用文本匹配: {str(e)}")
            results = self.db.query(AIBusinessKnowledge).filter(
                AIBusinessKnowledge.answer_text.ilike(f"%{query_text}%")
            ).limit(top_k).all()
            
            search_results = []
            for knowledge in results:
                result_dict = knowledge.to_dict()
                result_dict['similarity'] = 0.5  # 默认相似度
                search_results.append(result_dict)
            return search_results
    
    def count(self) -> int:
        """
        获取知识库中的总记录数
        
        Returns:
            总记录数
        """
        return self.db.query(AIBusinessKnowledge).count()

