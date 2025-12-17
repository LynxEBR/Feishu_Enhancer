"""
飞书任务增强器 - 联网搜索 + PostgreSQL 向量缓存
"""
import os
import re
import asyncio
import httpx
from typing import Optional, Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 导入业务知识库相关模块
from business_knowledge.database import get_db
from business_knowledge.crud import BusinessKnowledgeCRUD


class FeishuTaskEnhancer:
    """
    飞书任务增强器
    
    核心功能：
    1. 语义缓存（PostgreSQL 向量数据库）
    2. 智能检索（相似度 >= 阈值直接返回）
    3. 联网搜索（Bocha API，限定飞书官网）
    4. 自动沉淀（搜索结果自动入库）
    """
    
    # 飞书官方文档 URL 模式（白名单）
    FEISHU_OFFICIAL_URL_PATTERNS = [
        'https://www.feishu.cn/hc/zh-CN/articles/',
        'https://www.feishu.cn/hc/en-US/articles/',
        'https://www.feishu.cn/hc/zh-CN/',
        'https://www.feishu.cn/hc/en-US/',
        'https://www.larksuite.com/hc/',
        'https://feishu.cn/hc/',
    ]
    
    def __init__(
        self,
        llm: ChatOpenAI,
        bocha_api_key: Optional[str] = None,
        bocha_base_url: str = "https://api.bochaai.com/v1",
        similarity_threshold: float = 0.85,
        enable_cache: bool = True,
        enable_web_search: bool = True,
        verbose: bool = True,
        max_search_iterations: int = 2
    ):
        """
        初始化增强器
        
        Args:
            llm: LangChain ChatOpenAI 实例
            bocha_api_key: Bocha API 密钥（用于联网搜索）
            bocha_base_url: Bocha API 基础 URL
            similarity_threshold: 缓存命中的相似度阈值
            enable_cache: 是否启用缓存
            enable_web_search: 是否启用联网搜索
            verbose: 是否输出详细日志
            max_search_iterations: 最大搜索迭代次数
        """
        self.llm = llm
        self.bocha_api_key = bocha_api_key or os.getenv("BOCHA_API_KEY", "")
        self.bocha_base_url = bocha_base_url
        self.similarity_threshold = similarity_threshold
        self.enable_cache = enable_cache
        self.enable_web_search = enable_web_search
        self.verbose = verbose
        self.max_search_iterations = max_search_iterations
        
        if self.verbose:
            print(f"✓ FeishuTaskEnhancer 初始化完成")
            print(f"  - 相似度阈值: {self.similarity_threshold}")
            print(f"  - 缓存启用: {self.enable_cache}")
            print(f"  - 联网搜索启用: {self.enable_web_search}")
        
    def _get_crud(self) -> BusinessKnowledgeCRUD:
        """获取数据库 CRUD 实例"""
        db = next(get_db())
        return BusinessKnowledgeCRUD(db)
    
    def _is_feishu_official_doc(self, url: str) -> bool:
        """
        判断是否是飞书官方文档
        
        Args:
            url: 网页 URL
            
        Returns:
            是否为官方文档
        """
        return any(url.startswith(pattern) for pattern in self.FEISHU_OFFICIAL_URL_PATTERNS)
    
    def _score_feishu_result(self, page: Dict, query: str) -> Tuple[bool, int]:
        """
        对搜索结果进行评分
        
        评分规则：
        - 官方文档 (feishu.cn/hc/): +100 分
        - 标题包含查询词: +10 分
        - 摘要包含查询词: +5 分
        - URL 包含 /hc/: +20 分
        - 摘要长度适中 (50-500): +3 分
        
        Args:
            page: 搜索结果页面信息
            query: 原始查询
            
        Returns:
            (是否官方文档, 评分)
        """
        url = page.get('url', '')
        name = page.get('name', '').lower()
        summary = page.get('summary', page.get('snippet', '')).lower()
        query_lower = query.lower()
        
        is_official = self._is_feishu_official_doc(url)
        score = 0
        
        # 官方文档加分
        if is_official:
            score += 100
        
        # 标题匹配
        if query_lower in name:
            score += 10
        
        # 摘要匹配
        if query_lower in summary:
            score += 5
        
        # 帮助中心路径加分
        if 'feishu.cn/hc' in url or '/hc/' in url:
            score += 20
        
        # 摘要长度适中加分
        summary_len = len(summary)
        if 50 < summary_len < 500:
            score += 3
        
        return (is_official, score)
    
    def _bocha_feishu_search_sync(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        使用 Bocha API 搜索飞书官方文档（同步版本）
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 【关键】强制限定飞书官方域名 + 帮助中心
        search_query = f"site:feishu.cn 帮助中心 {query}"
        
        if self.verbose:
            print(f"[联网搜索] 搜索关键词: {search_query}")
        
        if not self.bocha_api_key:
            print("[联网搜索] 未配置 Bocha API Key，跳过联网搜索")
            return []
        
        import requests
        
        url = f"{self.bocha_base_url}/web-search"
        headers = {
            'Authorization': f'Bearer {self.bocha_api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "query": search_query,
            "freshness": "noLimit",
            "summary": True,
            "count": 30  # 多取一些，用于筛选
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                print(f"[联网搜索] API 返回错误: {response.status_code}")
                return []
            
            json_response = response.json()
            if json_response.get("code") != 200 or not json_response.get("data"):
                print(f"[联网搜索] API 返回错误: {json_response.get('msg', '未知错误')}")
                return []
            
            webpages = json_response["data"].get("webPages", {}).get("value", [])
            
            if not webpages:
                return []
            
            # 【关键】二次清洗与评分
            official_docs = []  # 官方帮助中心文档
            other_feishu = []   # 其他飞书页面
            
            for page in webpages:
                url_str = page.get('url', '')
                
                if self._is_feishu_official_doc(url_str):
                    _, score = self._score_feishu_result(page, query)
                    official_docs.append((page, score))
                elif 'feishu.cn' in url_str.lower() or 'larksuite.com' in url_str.lower():
                    _, score = self._score_feishu_result(page, query)
                    other_feishu.append((page, score))
            
            # 按评分排序
            official_docs.sort(key=lambda x: -x[1])
            other_feishu.sort(key=lambda x: -x[1])
            
            # 优先返回官方文档，不足则补充其他飞书页面
            results = [p for p, _ in official_docs[:count]]
            if len(results) < count and other_feishu:
                results.extend([p for p, _ in other_feishu[:count - len(results)]])
            
            print(f"[联网搜索] 获取到 {len(results)} 条有效结果（其中官方文档 {len(official_docs)} 条）")
            return results
            
        except Exception as e:
            print(f"[联网搜索] 搜索失败: {str(e)}")
            return []
    
    async def _bocha_feishu_search(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        使用 Bocha API 搜索飞书官方文档（异步版本）
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 【关键】强制限定飞书官方域名 + 帮助中心
        search_query = f"site:feishu.cn 帮助中心 {query}"
        
        if self.verbose:
            print(f"[联网搜索] 搜索关键词: {search_query}")
        
        if not self.bocha_api_key:
            print("[联网搜索] 未配置 Bocha API Key，跳过联网搜索")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.bocha_base_url}/web-search",
                    headers={
                        "Authorization": f"Bearer {self.bocha_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": search_query,
                        "summary": True,
                        "count": 30,
                        "freshness": "noLimit"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    webpages = data.get("data", {}).get("webPages", {}).get("value", [])
                    
                    # 二次清洗与评分
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
                    
                    # 排序
                    official_docs.sort(key=lambda x: -x[1])
                    other_feishu.sort(key=lambda x: -x[1])
                    
                    # 合并结果
                    results = [p for p, _ in official_docs[:count]]
                    if len(results) < count:
                        results.extend([p for p, _ in other_feishu[:count - len(results)]])
                    
                    print(f"[联网搜索] 获取到 {len(results)} 条有效结果（其中官方文档 {len(official_docs)} 条）")
                    return results
                else:
                    print(f"[联网搜索] API 返回错误: {response.status_code}")
                    return []
                    
        except Exception as e:
            print(f"[联网搜索] 搜索失败: {str(e)}")
            return []
    
    async def search_cache(self, question: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        在 PostgreSQL 向量库中搜索相似问题
        
        Args:
            question: 查询问题
            top_k: 返回结果数量
            
        Returns:
            如果找到相似度超过阈值的结果，返回最相似的结果；否则返回 None
        """
        if not self.enable_cache:
            return None
            
        try:
            crud = self._get_crud()
            results = crud.search_by_question(
                query_text=question,
                top_k=top_k,
                threshold=self.similarity_threshold
            )
            
            if results and len(results) > 0:
                best_match = results[0]
                if self.verbose:
                    print(f"[缓存查询] 找到相似问题，相似度: {best_match.get('similarity', 0):.4f}")
                    print(f"[缓存查询] 原问题: {best_match.get('question_text', '')[:100]}...")
                return best_match
            else:
                if self.verbose:
                    print(f"[缓存查询] 未找到相似度 >= {self.similarity_threshold} 的缓存结果")
                return None
                
        except Exception as e:
            print(f"[缓存查询] 搜索失败: {str(e)}")
            return None
    
    async def save_to_cache(self, question: str, answer: str) -> bool:
        """
        将问答对存入 PostgreSQL 向量库
        
        Args:
            question: 问题文本（原始任务）
            answer: 答案文本（增强后的任务描述）
            
        Returns:
            是否保存成功
        """
        if not self.enable_cache:
            return False
            
        try:
            crud = self._get_crud()
            knowledge = crud.create(
                question_text=question,
                answer_text=answer
            )
            if self.verbose:
                print(f"[缓存保存] 成功保存知识条目，ID: {knowledge.id}")
            return True
        except Exception as e:
            print(f"[缓存保存] 保存失败: {str(e)}")
            return False
    
    async def _generate_search_keywords(self, task: str) -> List[str]:
        """
        使用 LLM 生成搜索关键词
        
        Args:
            task: 原始任务描述
            
        Returns:
            搜索关键词列表
        """
        system_prompt = """你是一个飞书产品专家。根据用户的测试任务，生成用于搜索飞书帮助文档的关键词。

要求：
1. 提取任务中的核心功能点
2. 使用飞书产品术语
3. 返回 2-3 个搜索关键词，每行一个
4. 只返回关键词，不要其他解释"""

        user_prompt = f"任务: {task}\n\n请生成搜索关键词："

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            keywords = [kw.strip() for kw in response.content.strip().split('\n') if kw.strip()]
            return keywords[:3]  # 最多返回3个关键词
        except Exception as e:
            print(f"[关键词生成] 失败: {str(e)}")
            # 回退：直接使用原任务作为关键词
            return [task]
    
    async def _generate_enhanced_task(
        self,
        original_task: str,
        search_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        使用 LLM 根据搜索结果生成增强任务
        
        Args:
            original_task: 原始任务描述
            search_results: 联网搜索结果
            
        Returns:
            (增强后的任务描述, 引用的URL列表)
        """
        # 整理搜索结果
        search_context = ""
        source_urls = []
        
        if search_results:
            search_snippets = []
            for i, result in enumerate(search_results[:5], 1):
                title = result.get("name", "")
                snippet = result.get("summary", result.get("snippet", ""))
                url = result.get("url", "")
                is_official = "★ 官方文档" if self._is_feishu_official_doc(url) else ""
                search_snippets.append(f"{i}. {is_official} {title}\n   摘要: {snippet}\n   链接: {url}")
                source_urls.append(url)
            search_context = "\n\n".join(search_snippets)
        
        system_prompt = """你是一个飞书PC桌面端UI自动化测试工程师。你的职责是：
1. 根据提供的飞书官方文档搜索结果，补全和优化用户的测试任务描述
2. 将模糊的概念替换为具体、明确的PC端UI操作步骤

【四大原则】：

1. **单人视角原则（核心）**
   - 场景中只能存在"当前用户"一个操作者
   - 严禁描述接收方的主动行为（如"接收方点击"、"等待对方回复"）
   - 所有验证点必须是当前用户界面上可见的内容

2. **动态目标策略**
   - 如果用户原始测试点中指定了接收对象，在步骤中明确描述UI选择操作
   - 如果未指定对象，默认操作对象为"文件传输助手"或"当前用户自己"
   - 将人视为UI元素：选择联系人的过程应描述为UI操作

3. **PC端交互规范**
   - 使用PC桌面端术语：鼠标左键/右键点击、多选消息、侧边栏、新窗口打开等
   - 描述完整链路：选中 -> 操作 -> 验证
   - 明确操作入口

4. **自我验证原则**
   - 验证点必须在当前用户的界面上
   - 不依赖外部反馈，所有验证都是对UI元素状态的检查

输出要求：
- 输出清晰、可执行的测试步骤
- 使用PC端术语
- 不添加额外的解释，只输出步骤
- 不要输出 URL 链接"""

        user_prompt = f"""请将以下测试点补全为详细的PC端UI操作步骤：

【原始测试点】
{original_task}

【飞书官方文档参考】
{search_context if search_context else "无搜索结果"}

【补全后的测试步骤】"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            enhanced_task = response.content.strip()
            
            # 清理结果中可能残留的 URL
            enhanced_task = re.sub(r'https?://\S+', '', enhanced_task)
            enhanced_task = re.sub(r'\s+', ' ', enhanced_task).strip()
            
            return enhanced_task, source_urls
        except Exception as e:
            print(f"[LLM生成] 生成失败: {str(e)}")
            return original_task, []
    
    async def _execute_search_and_enhance(self, task: str) -> Tuple[str, List[str]]:
        """
        执行搜索和增强流程
        
        流程：
        1. 生成搜索关键词
        2. 执行联网搜索
        3. 使用 LLM 生成增强任务
        
        Args:
            task: 原始任务描述
            
        Returns:
            (增强后的任务描述, 引用的URL列表)
        """
        all_search_results = []
        
        # Step 1: 生成搜索关键词
        if self.verbose:
            print("[搜索增强] 生成搜索关键词...")
        keywords = await self._generate_search_keywords(task)
        if self.verbose:
            print(f"[搜索增强] 关键词: {keywords}")
        
        # Step 2: 执行搜索
        for keyword in keywords:
            if self.verbose:
                print(f"[搜索增强] 搜索: {keyword}")
            results = await self._bocha_feishu_search(keyword, count=3)
            all_search_results.extend(results)
        
        # 去重（基于 URL）
        seen_urls = set()
        unique_results = []
        for result in all_search_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        if self.verbose:
            print(f"[搜索增强] 共获取 {len(unique_results)} 条唯一结果")
        
        # Step 3: 使用 LLM 生成增强任务
        if self.verbose:
            print("[搜索增强] 生成增强任务...")
        enhanced_task, source_urls = await self._generate_enhanced_task(task, unique_results)
        
        return enhanced_task, source_urls
    
    async def enhance(self, task: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        增强任务描述的主入口
        
        流程：
        1. 搜索缓存（PostgreSQL 向量库）
        2. 缓存命中 -> 直接返回缓存的答案
        3. 缓存未命中 -> 联网搜索 -> LLM 生成 -> 存入缓存 -> 返回结果
        
        Args:
            task: 原始任务描述
            force_refresh: 是否强制刷新（跳过缓存）
            
        Returns:
            包含增强结果的字典：
            - enhanced_task: 增强后的任务描述
            - cache_hit: 是否命中缓存
            - search_performed: 是否执行了联网搜索
            - source: 结果来源 ("cache" / "search" / "original")
            - source_urls: 引用的URL列表
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📝 任务增强开始: {task[:100]}...")
            print(f"{'='*60}")
        
        result = {
            "original_task": task,
            "enhanced_task": task,
            "cache_hit": False,
            "search_performed": False,
            "source": "original",
            "source_urls": []
        }
        
        # Step 1: 搜索缓存（除非强制刷新）
        if not force_refresh and self.enable_cache:
            if self.verbose:
                print("\n[Step 1] 搜索向量缓存...")
            cache_result = await self.search_cache(task)
            
            if cache_result:
                if self.verbose:
                    print(f"✓ 命中缓存（相似度: {cache_result.get('similarity', 0):.4f}）")
                result["enhanced_task"] = cache_result.get("answer_text", task)
                result["cache_hit"] = True
                result["source"] = "cache"
                result["cache_similarity"] = cache_result.get("similarity", 0)
                return result
        else:
            if self.verbose:
                if force_refresh:
                    print("⚡ 强制刷新模式，跳过缓存")
                else:
                    print("⚠️ 缓存已禁用")
        
        # Step 2: 缓存未命中，执行联网搜索 + LLM 生成
        if self.enable_web_search:
            if self.verbose:
                print("\n[Step 2] 启动联网搜索...")
            
            enhanced_task, source_urls = await self._execute_search_and_enhance(task)
            
            result["enhanced_task"] = enhanced_task
            result["search_performed"] = True
            result["source"] = "search"
            result["source_urls"] = source_urls
        else:
            if self.verbose:
                print("\n⚠️ 联网搜索已禁用，使用原始任务")
        
        # Step 3: 将结果存入缓存
        if result["enhanced_task"] != task and self.enable_cache:
            if self.verbose:
                print("\n[Step 3] 保存到向量缓存...")
            await self.save_to_cache(task, result["enhanced_task"])
        
        if self.verbose:
            print(f"\n✓ 任务增强完成")
            print(f"{'='*60}\n")
        
        return result


class FeishuEnhancerConfig:
    """飞书增强器配置类"""
    
    def __init__(self):
        # 从环境变量或配置文件加载
        try:
            import config
            self.bocha_api_key = config.config_dict.get("BOCHA_API_KEY", os.getenv("BOCHA_API_KEY", ""))
            self.bocha_base_url = config.config_dict.get("BOCHA_BASE_URL", "https://api.bochaai.com/v1")
            self.similarity_threshold = float(config.config_dict.get("ENHANCE_SIMILARITY_THRESHOLD", "0.85"))
            self.enable_cache = str(config.config_dict.get("ENHANCE_ENABLE_CACHE", "true")).lower() == "true"
            self.enable_web_search = str(config.config_dict.get("ENHANCE_ENABLE_WEB_SEARCH", "true")).lower() == "true"
        except ImportError:
            self.bocha_api_key = os.getenv("BOCHA_API_KEY", "")
            self.bocha_base_url = os.getenv("BOCHA_BASE_URL", "https://api.bochaai.com/v1")
            self.similarity_threshold = float(os.getenv("ENHANCE_SIMILARITY_THRESHOLD", "0.85"))
            self.enable_cache = os.getenv("ENHANCE_ENABLE_CACHE", "true").lower() == "true"
            self.enable_web_search = os.getenv("ENHANCE_ENABLE_WEB_SEARCH", "true").lower() == "true"


# 全局增强器实例（延迟初始化）
_enhancer_instance: Optional[FeishuTaskEnhancer] = None


def get_feishu_enhancer(llm: ChatOpenAI) -> FeishuTaskEnhancer:
    """
    获取飞书增强器实例（单例模式）
    
    Args:
        llm: LangChain ChatOpenAI 实例
        
    Returns:
        FeishuTaskEnhancer 实例
    """
    global _enhancer_instance
    if _enhancer_instance is None:
        config = FeishuEnhancerConfig()
        _enhancer_instance = FeishuTaskEnhancer(
            llm=llm,
            bocha_api_key=config.bocha_api_key,
            bocha_base_url=config.bocha_base_url,
            similarity_threshold=config.similarity_threshold,
            enable_cache=config.enable_cache,
            enable_web_search=config.enable_web_search
        )
    return _enhancer_instance