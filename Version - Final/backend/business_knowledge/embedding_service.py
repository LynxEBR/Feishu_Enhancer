"""
Embedding 服务 - 使用 BGE-Large 模型（通过 transformers）
"""
import numpy as np
import torch
from typing import List, Union
from transformers import AutoModel, AutoTokenizer


class EmbeddingService:
    """BGE-Large 模型 Embedding 服务（使用 transformers）"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化 Embedding 服务
        
        Args:
            model_name: BGE 模型名称，默认为 bge-large-zh-v1.5
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """加载 BGE 模型"""
        try:
            print(f"正在加载 BGE 模型: {self.model_name}")
            print(f"使用设备: {self.device}")
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # 设置为评估模式
            
            # 移动到指定设备
            self.model = self.model.to(self.device)
            
            print(f"BGE 模型加载成功: {self.model_name}")
        except Exception as e:
            print(f"加载 BGE 模型失败: {str(e)}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        平均池化，获取句子嵌入
        
        Args:
            model_output: 模型输出
            attention_mask: 注意力掩码
            
        Returns:
            池化后的嵌入向量
        """
        # 获取 token embeddings
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        
        # 扩展 attention_mask 的维度以匹配 token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 计算加权和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # 计算实际 token 数量（避免除以零）
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 返回平均池化结果
        return sum_embeddings / sum_mask
    
    def encode_question(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        编码问题文本为向量
        
        Args:
            text: 单个问题文本或问题文本列表
            normalize: 是否归一化向量
            
        Returns:
            向量数组，形状为 (n, 1024) 或 (1024,)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")
        
        # 如果是单个字符串，转换为列表
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # Tokenize
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移动到设备
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # 生成嵌入
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 归一化
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为 numpy 数组
        embeddings = embeddings.cpu().numpy()
        
        # 如果是单个文本，返回一维数组
        if is_single:
            embeddings = embeddings[0]
        
        return embeddings
    
    def encode_answer(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        编码答案文本为向量
        
        Args:
            text: 单个答案文本或答案文本列表
            normalize: 是否归一化向量
            
        Returns:
            向量数组，形状为 (n, 1024) 或 (1024,)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")
        
        # 如果是单个字符串，转换为列表
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # 答案不需要添加查询指令
        # Tokenize
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 移动到设备
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # 生成嵌入
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 归一化
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为 numpy 数组
        embeddings = embeddings.cpu().numpy()
        
        # 如果是单个文本，返回一维数组
        if is_single:
            embeddings = embeddings[0]
        
        return embeddings
    
    def encode(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        通用编码方法
        
        Args:
            text: 文本或文本列表
            normalize: 是否归一化向量
            
        Returns:
            向量数组
        """
        return self.encode_question(text, normalize=normalize)


# 全局 Embedding 服务实例
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """获取全局 Embedding 服务实例（单例模式）"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

