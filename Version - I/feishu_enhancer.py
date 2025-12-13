"""
飞书测试用代码
基于 Chroma 语义缓存的 RAG 系统

使用示例：
    # 同步方式
    enhancer = FeishuTestCaseEnhancer()
    result = enhancer.enhance("云文档被分享到IM中能正常打开")
    
    # 异步方式
    enhancer = FeishuTestCaseEnhancer()
    result = await enhancer.aenhance("云文档被分享到IM中能正常打开")
"""

import os
import re
import asyncio
from typing import Dict, List, Optional
from openai import OpenAI, AsyncOpenAI
import chromadb
from chromadb.config import Settings
import requests
from urllib.parse import urlparse

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI


class FeishuTestCaseEnhancer:
    """
    飞书测试用例增强器
    
    核心功能：
    1. 语义缓存（Chroma 向量数据库）
    2. 智能检索（相似度 >= 0.90 直接返回）
    3. Agent 搜索（LangChain + Bocha API）
    4. 自动沉淀（搜索结果自动入库）
    """
    
    def __init__(
        self,
        dashscope_api_key: str = None,
        bocha_api_key: str = None,
        chroma_persist_dir: str = "./chroma_db",
        chroma_collection_name: str = "feishu_test_cases",
        similarity_threshold: float = 0.90,
        embedding_model: str = "text-embedding-v3",
        embedding_dimensions: int = 1024,
        llm_model: str = "deepseek-v3",
        verbose: bool = True
    ):
        """
        初始化增强器
        
        Args:
            dashscope_api_key: 阿里云百炼 API 密钥（用于 Embedding 和 LLM）
            bocha_api_key: Bocha 搜索 API 密钥
            chroma_persist_dir: Chroma 数据库持久化目录
            chroma_collection_name: 集合名称
            similarity_threshold: 相似度阈值（0-1）
            embedding_model: Embedding 模型名称
            embedding_dimensions: 向量维度
            llm_model: LLM 模型名称
            verbose: 是否输出详细日志
        """
        # API 密钥配置
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
        self.bocha_api_key = bocha_api_key or "sk-c916a733feab496bb15c8b429b669709"
        
        if not self.dashscope_api_key:
            raise ValueError("必须提供 DASHSCOPE_API_KEY（通过参数或环境变量）")
        
        # 配置参数
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.llm_model = llm_model
        self.verbose = verbose
        
        # 飞书官方文档 URL 模式
        self.feishu_official_url_patterns = [
            'https://www.feishu.cn/hc/zh-CN/articles/',
            'https://www.feishu.cn/hc/en-US/articles/',
            'https://www.larksuite.com/hc/',
        ]
        
        # 初始化 OpenAI 客户端（用于 Embedding）
        self.openai_client = OpenAI(
            api_key=self.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化异步 OpenAI 客户端
        self.async_openai_client = AsyncOpenAI(
            api_key=self.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化 Chroma 向量数据库
        self._init_chroma(chroma_persist_dir, chroma_collection_name)
        
        # 初始化 LangChain LLM
        self.llm = ChatOpenAI(
            api_key=self.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=self.llm_model,
            temperature=0
        )
        
        # 创建搜索工具
        self._init_search_tool()
        
        if self.verbose:
            print(f"FeishuTestCaseEnhancer 初始化完成")
            print(f"  - 向量库: {chroma_persist_dir}/{chroma_collection_name}")
            print(f"  - 缓存数量: {self.collection.count()}")
            print(f"  - 相似度阈值: {self.similarity_threshold}")
    
    def _init_chroma(self, persist_dir: str, collection_name: str):
        """初始化 Chroma 向量数据库"""
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "飞书UI测试用例缓存"}
        )
    
    def _init_search_tool(self):
        """初始化飞书文档搜索工具"""
        self.feishu_search_tool = Tool(
            name="FeishuDocSearch",
            func=lambda query: self._bocha_feishu_search(query, count=5),
            description="专门搜索飞书官方帮助文档(feishu.cn/hc)。输入应为搜索查询字符串，输出将优先返回飞书官方文档的搜索结果。"
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成向量（同步）"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    async def _agenerate_embedding(self, text: str) -> List[float]:
        """生成向量（异步）"""
        response = await self.async_openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def _query_cache(self, query_embedding: List[float]) -> Optional[Dict]:
        """查询缓存"""
        if self.collection.count() == 0:
            if self.verbose:
                print("向量库为空，无缓存可查")
            return None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['ids'][0]:
            if self.verbose:
                print("未找到相似结果")
            return None
        
        distance = results['distances'][0][0]
        similarity = 1 - distance
        
        if self.verbose:
            print(f"最高相似度: {similarity:.4f} (阈值: {self.similarity_threshold})")
        
        if similarity >= self.similarity_threshold:
            if self.verbose:
                print("命中缓存！")
            return {
                'test_case': results['documents'][0][0],
                'metadata': results['metadatas'][0][0],
                'similarity': similarity,
                'id': results['ids'][0][0]
            }
        else:
            if self.verbose:
                print(f"相似度不足（{similarity:.4f} < {self.similarity_threshold}）")
            return None
    
    def _add_to_cache(self, query: str, query_embedding: List[float], 
                     test_case: str, metadata: Dict):
        """添加到缓存"""
        import uuid
        from datetime import datetime
        
        doc_id = str(uuid.uuid4())
        metadata['original_query'] = query
        metadata['cached_at'] = datetime.now().isoformat()
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[query_embedding],
            documents=[test_case],
            metadatas=[metadata]
        )
        
        if self.verbose:
            print(f"已存入缓存: {doc_id}")
            print(f"当前缓存总量: {self.collection.count()}")
    
    def _is_feishu_official_doc(self, url: str) -> bool:
        """判断是否是飞书官方文档"""
        return any(url.startswith(pattern) for pattern in self.feishu_official_url_patterns)
    
    def _score_feishu_result(self, page: dict, query: str) -> tuple:
        """对搜索结果评分"""
        url = page.get('url', '')
        name = page.get('name', '').lower()
        summary = page.get('summary', '').lower()
        query_lower = query.lower()
        
        is_official = self._is_feishu_official_doc(url)
        score = 0
        
        if is_official:
            score += 100
        if query_lower in name:
            score += 10
        if query_lower in summary:
            score += 5
        if 'feishu.cn/hc' in url:
            score += 20
        
        summary_len = len(summary)
        if 50 < summary_len < 500:
            score += 3
        
        return (is_official, score)
    
    def _bocha_feishu_search(self, query: str, count: int = 5) -> str:
        """搜索飞书官方文档"""
        search_query = f"site:feishu.cn 帮助中心 {query}"
        
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer {self.bocha_api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "query": search_query,
            "freshness": "noLimit",
            "summary": True,
            "count": 30
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                return f"搜索API请求失败，状态码: {response.status_code}"
            
            json_response = response.json()
            if json_response.get("code") != 200 or not json_response.get("data"):
                return f"搜索API返回错误: {json_response.get('msg', '未知错误')}"
            
            webpages = json_response["data"].get("webPages", {}).get("value", [])
            
            if not webpages:
                return "未找到相关结果"
            
            official_docs = []
            other_feishu = []
            
            for page in webpages:
                url_str = page.get('url', '')
                
                if self._is_feishu_official_doc(url_str):
                    _, score = self._score_feishu_result(page, query)
                    official_docs.append((page, score))
                elif 'feishu.cn' in url_str.lower() or 'larksuite.com' in url_str.lower():
                    _, score = self._score_feishu_result(page, query)
                    other_feishu.append((page, score))
            
            official_docs.sort(key=lambda x: -x[1])
            other_feishu.sort(key=lambda x: -x[1])
            
            if not official_docs and not other_feishu:
                return f"未找到飞书官方文档"
            
            results_to_show = official_docs[:count]
            if len(results_to_show) < count and other_feishu:
                results_to_show.extend(other_feishu[:count - len(results_to_show)])
            
            formatted_results = ""
            if official_docs:
                formatted_results += f"✓ 找到 {len(official_docs)} 个飞书官方文档\n"
            formatted_results += "\n"
            
            for idx, (page, score) in enumerate(results_to_show, start=1):
                is_official = self._is_feishu_official_doc(page['url'])
                tag = "★ 官方文档" if is_official else "飞书相关"
                
                formatted_results += (
                    f"[{idx}] {tag}\n"
                    f"标题: {page['name']}\n"
                    f"链接: {page['url']}\n"
                    f"说明: {page.get('summary', '无摘要')}\n"
                    f"{'-'*60}\n\n"
                )
            
            return formatted_results.strip()
            
        except Exception as e:
            return f"搜索过程出错: {str(e)}"
    
    def _execute_agent_search(self, user_question: str) -> tuple[str, List[str]]:
        """执行 Agent 搜索"""
        agent = initialize_agent(
            tools=[self.feishu_search_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        prompt = f"""
你现在的身份是：**飞书PC桌面端（Windows/macOS）UI自动化测试工程师**。
你的运行环境是：**单台电脑、单个测试账号**。

任务：根据用户提供的测试点，利用 FeishuDocSearch 搜索官方文档，将其补全为一条**仅当前PC客户端用户可见可操作**的测试步骤描述。

用户原始测试点: "{user_question}"

请严格遵守以下【四大原则】：

1. **单人视角原则（核心）**
   - 场景中只能存在"当前用户"一个操作者。
   - **严禁**描述接收方的主动行为（如"接收方点击"、"等待对方回复"）。

2. **动态目标策略**
    - **优先提取**：如果用户原始测试点中指定了接收对象（如"转发给张三"、"分享到测试群"），请在步骤中明确描述"在联系人选择器中搜索并选中**'张三'**"。
   - **默认兜底**：如果原始测试点未指定对象（仅说"转发消息"），则默认操作对象为**"文件传输助手"**或**"当前用户自己"**。
   - **将人视为UI元素**：选择联系人的过程应描述为UI操作（搜索、点击勾选、确认）。

3. **PC端交互规范**
   - 使用PC桌面端术语：如"鼠标左键/右键点击"、"多选消息"、"合并转发"、"侧边栏"、"新窗口打开"。
   - 描述完整链路：选中消息 -> 点击转发 -> **搜索并选中目标（指定用户或默认）** -> 点击发送 -> **自我验证**。

4. **自我验证原则**
   - 验证点必须在当前用户的界面上。
   - 示例："发送后，当前用户在会话窗口中点击刚才发送的【合并转发】卡片，验证能否展开详情并打开其中的云文档"。

---
**执行步骤**:
1. Search: 搜索功能在PC端的入口（如：右键菜单中的转发）和UI表现。
2. Transform: 将"发给某人"转换为"UI上选择某个联系人"。
3. Write: 输出符合单机操作逻辑的详细步骤。

---

现在，请开始执行任务。
请记住：**选择联系人只是一个UI点击动作，不需要对方真的在线或回应**。
"""
        
        source_urls = []
        try:
            result = agent.run(prompt)
            
            # 提取 URL
            url_pattern = r'https?://[^\s\[\]`"]+'
            source_urls = list(set(re.findall(url_pattern, result)))
            
            # 清理结果
            if "Final Answer:" in result:
                result = result.split("Final Answer:")[-1].strip()
            
            result = re.sub(r'https?://\S+', '', result)
            result = re.sub(r'\[官方文档\]|\[飞书相关\]|\[[0-9]+\]|引用\s*\d+|★', '', result)
            result = re.sub(r'\n+', ' ', result)
            result = re.sub(r'\s+', ' ', result).strip()
            result = re.sub(r'(Thought:|Action:|Action Input:|Observation:).*', '', result, flags=re.IGNORECASE)
            
            return result, source_urls
            
        except Exception as e:
            error_msg = str(e)
            
            if "Could not parse LLM output:" in error_msg:
                matches = re.findall(r'`([^`]+)`', error_msg)
                if matches:
                    result = max(matches, key=len)
                    
                    if len(result) < 50:
                        if "Could not parse LLM output:" in error_msg:
                            output_part = error_msg.split("Could not parse LLM output:")[-1]
                            output_part = output_part.strip(' :`"')
                            if len(output_part) > len(result):
                                result = output_part
                    
                    result = re.sub(r'^(Thought:|Action:|Action Input:|Observation:|Final Answer:)\s*', '', result, flags=re.IGNORECASE)
                    result = re.sub(r'(Thought:|Action:|Action Input:|Observation:).*$', '', result, flags=re.IGNORECASE)
                    result = re.sub(r'https?://\S+', '', result)
                    result = re.sub(r'\[官方文档\]|\[飞书相关\]|\[[0-9]+\]|引用\s*\d+|★', '', result)
                    result = re.sub(r'\s*[`"\'\n]+\s*', ' ', result)
                    result = re.sub(r'\s+', ' ', result).strip()
                    
                    if 30 <= len(result) <= 1000:
                        return result, source_urls
            
            return f"补全失败。建议: 1) 简化查询关键词 2) 检查网络连接 3) 稍后重试。错误详情: {error_msg[:100]}", []
    
    def enhance(self, question: str, force_refresh: bool = False) -> str:
        """
        补全测试用例（同步方法）
        
        Args:
            question: 原始测试点描述
            force_refresh: 是否强制刷新（跳过缓存）
        
        Returns:
            补全后的测试用例文本
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"用户问题: {question}")
            print(f"{'='*60}")
        
        # 生成向量
        if self.verbose:
            print("正在生成向量...")
        query_embedding = self._generate_embedding(question)
        
        # 查询缓存
        if not force_refresh:
            if self.verbose:
                print("查询向量库...")
            cached_result = self._query_cache(query_embedding)
            
            if cached_result:
                if self.verbose:
                    print(f"命中缓存（相似度: {cached_result['similarity']:.4f}）")
                return cached_result['test_case']
        else:
            if self.verbose:
                print("强制刷新模式，跳过缓存")
        
        # 未命中 - 执行 Agent 搜索
        if self.verbose:
            print("启动 Agent 搜索...")
        
        test_case, source_urls = self._execute_agent_search(question)
        
        # 存入缓存
        if self.verbose:
            print("写入向量库...")
        
        metadata = {
            'source_urls': ','.join(source_urls) if source_urls else 'N/A',
            'generation_method': 'agent_search'
        }
        self._add_to_cache(question, query_embedding, test_case, metadata)
        
        return test_case
    
    async def aenhance(self, question: str, force_refresh: bool = False) -> str:
        """
        补全测试用例（异步方法）
        
        Args:
            question: 原始测试点描述
            force_refresh: 是否强制刷新（跳过缓存）
        
        Returns:
            补全后的测试用例文本
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"用户问题: {question}")
            print(f"{'='*60}")
        
        # 生成向量（异步）
        if self.verbose:
            print("正在生成向量...")
        query_embedding = await self._agenerate_embedding(question)
        
        # 查询缓存
        if not force_refresh:
            if self.verbose:
                print("查询向量库...")
            cached_result = self._query_cache(query_embedding)
            
            if cached_result:
                if self.verbose:
                    print(f"命中缓存（相似度: {cached_result['similarity']:.4f}）")
                return cached_result['test_case']
        else:
            if self.verbose:
                print("强制刷新模式，跳过缓存")
        
        # 未命中 - 执行 Agent 搜索（在线程池中运行同步代码）
        if self.verbose:
            print("启动 Agent 搜索...")
        
        # 使用 asyncio.to_thread 在线程池中运行同步的 Agent
        test_case, source_urls = await asyncio.to_thread(
            self._execute_agent_search, question
        )
        
        # 存入缓存
        if self.verbose:
            print("写入向量库...")
        
        metadata = {
            'source_urls': ','.join(source_urls) if source_urls else 'N/A',
            'generation_method': 'agent_search'
        }
        self._add_to_cache(question, query_embedding, test_case, metadata)
        
        return test_case
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'total_count': self.collection.count(),
            'collection_name': self.collection.name,
            'similarity_threshold': self.similarity_threshold
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "飞书UI测试用例缓存"}
        )
        if self.verbose:
            print("✓ 缓存已清空")


# ==================== 便捷函数 ====================

def create_enhancer(
    dashscope_api_key: str = None,
    **kwargs
) -> FeishuTestCaseEnhancer:
    """
    创建增强器的便捷函数
    
    Args:
        dashscope_api_key: API 密钥
        **kwargs: 其他参数
    
    Returns:
        FeishuTestCaseEnhancer 实例
    """
    return FeishuTestCaseEnhancer(
        dashscope_api_key=dashscope_api_key,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例1: 同步方式
    print("="*80)
    print("示例1: 同步方式")
    print("="*80)
    
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=True
    )
    
    questions = [
        "云文档被分享到IM中能正常打开",
        "验证消息转发功能正常",
    ]
    
    for q in questions:
        result = enhancer.enhance(q)
        print(f"\n原始: {q}")
        print(f"补全: {result[:150]}...")
        print("-"*80)
    
    # 查看缓存统计
    stats = enhancer.get_cache_stats()
    print(f"\n缓存统计: {stats}")
    
    # 示例2: 异步方式
    print("\n" + "="*80)
    print("示例2: 异步方式")
    print("="*80)
    
    async def async_example():
        enhancer = FeishuTestCaseEnhancer(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            verbose=True
        )
        
        question = "测试文档协作编辑功能"
        result = await enhancer.aenhance(question)
        
        print(f"\n原始: {question}")
        print(f"补全: {result[:150]}...")
    
    asyncio.run(async_example())
