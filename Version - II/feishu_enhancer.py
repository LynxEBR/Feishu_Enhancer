"""
飞书任务增强器 - 联网搜索 + PostgreSQL 向量缓存
基于 LangChain Agent 的 ReAct 模式实现
"""
import os
import re
import asyncio
import httpx
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, Tool, AgentType

# 导入业务知识库相关模块
from business_knowledge.database import get_db
from business_knowledge.crud import BusinessKnowledgeCRUD


class FeishuTaskEnhancer:
    """
    飞书任务增强器（基于 LangChain Agent）
    
    核心功能：
    1. 语义缓存（PostgreSQL 向量数据库）
    2. 智能检索（相似度 >= 阈值直接返回）
    3. Agent 搜索（LangChain ReAct + Bocha API，限定飞书官网）
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
        max_agent_iterations: int = 5
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
            max_agent_iterations: Agent 最大迭代次数
        """
        self.llm = llm
        self.bocha_api_key = bocha_api_key or os.getenv("BOCHA_API_KEY", "")
        self.bocha_base_url = bocha_base_url
        self.similarity_threshold = similarity_threshold
        self.enable_cache = enable_cache
        self.enable_web_search = enable_web_search
        self.verbose = verbose
        self.max_agent_iterations = max_agent_iterations
        
        # 初始化飞书文档搜索工具
        self._init_search_tool()
        
        if self.verbose:
            print(f"✓ FeishuTaskEnhancer 初始化完成")
            print(f"  - 相似度阈值: {self.similarity_threshold}")
            print(f"  - 缓存启用: {self.enable_cache}")
            print(f"  - 联网搜索启用: {self.enable_web_search}")
        
    def _get_crud(self) -> BusinessKnowledgeCRUD:
        """获取数据库 CRUD 实例"""
        db = next(get_db())
        return BusinessKnowledgeCRUD(db)
    
    def _init_search_tool(self):
        """初始化飞书文档搜索工具（供 Agent 使用）"""
        self.feishu_search_tool = Tool(
            name="FeishuDocSearch",
            func=lambda query: self._bocha_feishu_search_sync(query, count=5),
            description="专门搜索飞书官方帮助文档(feishu.cn/hc)。输入应为搜索查询字符串，输出将优先返回飞书官方文档的搜索结果。"
        )
    
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
    
    def _bocha_feishu_search_sync(self, query: str, count: int = 5) -> str:
        """
        使用 Bocha API 搜索飞书官方文档（同步版本，供 Agent 使用）
        
        关键改进：
        1. 硬编码 site:feishu.cn + 帮助中心 约束
        2. 结果二次清洗与评分
        3. 优先返回官方文档
        
        Args:
            query: 搜索查询
            count: 返回结果数量
            
        Returns:
            格式化的搜索结果字符串
        """
        # 【关键】强制限定飞书官方域名 + 帮助中心
        search_query = f"site:feishu.cn 帮助中心 {query}"
        
        if self.verbose:
            print(f"[FeishuDocSearch] 搜索关键词: {search_query}")
        
        if not self.bocha_api_key:
            return "错误：未配置 Bocha API Key"
        
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
                return f"搜索 API 请求失败，状态码: {response.status_code}"
            
            json_response = response.json()
            if json_response.get("code") != 200 or not json_response.get("data"):
                return f"搜索 API 返回错误: {json_response.get('msg', '未知错误')}"
            
            webpages = json_response["data"].get("webPages", {}).get("value", [])
            
            if not webpages:
                return "未找到相关飞书文档"
            
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
            
            if not official_docs and not other_feishu:
                return "未找到飞书官方文档"
            
            # 优先返回官方文档，不足则补充其他飞书页面
            results_to_show = official_docs[:count]
            if len(results_to_show) < count and other_feishu:
                results_to_show.extend(other_feishu[:count - len(results_to_show)])
            
            # 格式化输出
            formatted_results = ""
            if official_docs:
                formatted_results += f"✓ 找到 {len(official_docs)} 个飞书官方帮助文档\n"
            formatted_results += "\n"
            
            for idx, (page, score) in enumerate(results_to_show, start=1):
                is_official = self._is_feishu_official_doc(page['url'])
                tag = "★ 官方文档" if is_official else "飞书相关"
                
                formatted_results += (
                    f"[{idx}] {tag}\n"
                    f"标题: {page.get('name', '无标题')}\n"
                    f"链接: {page.get('url', '')}\n"
                    f"说明: {page.get('summary', page.get('snippet', '无摘要'))}\n"
                    f"{'-'*60}\n\n"
                )
            
            return formatted_results.strip()
            
        except Exception as e:
            return f"搜索过程出错: {str(e)}"
    
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
    
    def _execute_agent_search(self, user_question: str) -> Tuple[str, List[str]]:
        """
        执行 Agent 搜索（ReAct 模式）
        
        【核心改进】使用 LangChain Agent 让模型自主决定搜索策略，
        可以多次搜索并整合结果。
        
        Args:
            user_question: 用户原始问题/任务
            
        Returns:
            (增强后的任务描述, 引用的URL列表)
        """
        # 初始化 Agent
        agent = initialize_agent(
            tools=[self.feishu_search_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=self.max_agent_iterations,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        # 【关键】专业化提示词 - 飞书 PC 端 UI 自动化测试工程师角色
        prompt = f"""
你现在的身份是：**飞书PC桌面端（Windows/macOS）UI自动化测试工程师**。
你的运行环境是：**单台电脑、单个测试账号**。

任务：根据用户提供的测试点，利用 FeishuDocSearch 搜索飞书官方帮助文档，将其补全为一条**仅当前PC客户端用户可见可操作**的测试步骤描述。

用户原始测试点: "{user_question}"

请严格遵守以下【四大原则】：

1. **单人视角原则（核心）**
   - 场景中只能存在"当前用户"一个操作者。
   - **严禁**描述接收方的主动行为（如"接收方点击"、"等待对方回复"）。
   - 所有验证点必须是当前用户界面上可见的内容。

2. **动态目标策略**
   - **优先提取**：如果用户原始测试点中指定了接收对象（如"转发给张三"、"分享到测试群"），请在步骤中明确描述"在联系人选择器中搜索并选中**'张三'**"。
   - **默认兜底**：如果原始测试点未指定对象（仅说"转发消息"），则默认操作对象为**"文件传输助手"**或**"当前用户自己"**。
   - **将人视为UI元素**：选择联系人的过程应描述为UI操作（搜索、点击勾选、确认）。

3. **PC端交互规范**
   - 使用PC桌面端术语：如"鼠标左键/右键点击"、"多选消息"、"合并转发"、"侧边栏"、"新窗口打开"。
   - 描述完整链路：选中消息 -> 点击转发 -> **搜索并选中目标（指定用户或默认）** -> 点击发送 -> **自我验证**。
   - 明确操作入口：如"在聊天界面右上角点击..."、"在侧边栏文档列表中..."

4. **自我验证原则**
   - 验证点必须在当前用户的界面上。
   - 示例："发送后，当前用户在会话窗口中点击刚才发送的【合并转发】卡片，验证能否展开详情并打开其中的云文档"。
   - 不依赖外部反馈，所有验证都是对UI元素状态的检查。

---
**执行步骤**:
1. Search: 使用 FeishuDocSearch 搜索功能在PC端的入口（如：右键菜单中的转发）和UI表现。如果第一次搜索结果不够完整，可以调整关键词再次搜索。
2. Transform: 将"发给某人"转换为"UI上选择某个联系人"。
3. Write: 输出符合单机操作逻辑的详细步骤。

---

现在，请开始执行任务。
请记住：**选择联系人只是一个UI点击动作，不需要对方真的在线或回应**。
请只输出最终的测试步骤描述，不要输出思考过程。
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
            
            # 移除 URL
            result = re.sub(r'https?://\S+', '', result)
            # 移除标记
            result = re.sub(r'\[官方文档\]|\[飞书相关\]|\[[0-9]+\]|引用\s*\d+|★', '', result)
            # 清理多余空白
            result = re.sub(r'\n+', '\n', result)
            result = re.sub(r'\s+', ' ', result).strip()
            # 移除 Agent 输出格式
            result = re.sub(r'(Thought:|Action:|Action Input:|Observation:).*', '', result, flags=re.IGNORECASE)
            
            return result, source_urls
            
        except Exception as e:
            error_msg = str(e)
            
            # 尝试从错误信息中提取有效输出
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
                    
                    # 清理
                    result = re.sub(r'^(Thought:|Action:|Action Input:|Observation:|Final Answer:)\s*', '', result, flags=re.IGNORECASE)
                    result = re.sub(r'(Thought:|Action:|Action Input:|Observation:).*$', '', result, flags=re.IGNORECASE)
                    result = re.sub(r'https?://\S+', '', result)
                    result = re.sub(r'\[官方文档\]|\[飞书相关\]|\[[0-9]+\]|引用\s*\d+|★', '', result)
                    result = re.sub(r'\s*[`"\'\n]+\s*', ' ', result)
                    result = re.sub(r'\s+', ' ', result).strip()
                    
                    if 30 <= len(result) <= 2000:
                        return result, source_urls
            
            return f"任务增强失败。原始任务: {user_question}。建议: 1) 简化任务描述 2) 检查网络连接。错误: {error_msg[:100]}", []
    
    async def generate_enhanced_task_with_llm(
        self,
        original_task: str,
        search_results: List[Dict[str, Any]],
        background_knowledge: str = ""
    ) -> str:
        """
        使用 LLM 直接整合搜索结果生成增强任务（备用方法，当 Agent 模式不可用时使用）
        
        Args:
            original_task: 原始任务描述
            search_results: 联网搜索结果
            background_knowledge: 背景知识
            
        Returns:
            增强后的任务描述
        """
        # 整理搜索结果
        search_context = ""
        if search_results:
            search_snippets = []
            for i, result in enumerate(search_results[:5], 1):
                title = result.get("name", "")
                snippet = result.get("summary", result.get("snippet", ""))
                url = result.get("url", "")
                is_official = "★ 官方文档" if self._is_feishu_official_doc(url) else ""
                search_snippets.append(f"{i}. {is_official} {title}\n   {snippet}")
            search_context = "\n".join(search_snippets)
        
        system_prompt = """你是一个飞书PC桌面端UI自动化测试工程师。你的职责是：
1. 根据提供的飞书官方文档搜索结果，补全和优化用户的测试任务描述
2. 将模糊的概念替换为具体、明确的PC端UI操作步骤
3. 遵循"单人视角原则"：只描述当前用户的操作，不描述接收方行为
4. 遵循"自我验证原则"：验证点必须在当前用户界面上可见

输出格式要求：
- 输出清晰、可执行的测试步骤
- 使用PC端术语（鼠标点击、右键菜单、侧边栏等）
- 不添加额外的解释，只输出步骤"""

        context_parts = []
        if search_context:
            context_parts.append(f"【飞书官方文档搜索结果】\n{search_context}")
        if background_knowledge:
            context_parts.append(f"【背景知识】\n{background_knowledge}")
        
        context = "\n\n".join(context_parts) if context_parts else "无额外上下文"
        
        user_prompt = f"""请将以下测试点补全为详细的PC端UI操作步骤：

【原始测试点】
{original_task}

【参考信息】
{context}

【补全后的测试步骤】"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            enhanced_task = response.content.strip()
            return enhanced_task
        except Exception as e:
            print(f"[LLM生成] 生成失败: {str(e)}")
            return original_task
    
    async def enhance(self, task: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        增强任务描述的主入口
        
        流程：
        1. 搜索缓存（PostgreSQL 向量库）
        2. 缓存命中 -> 直接返回缓存的答案
        3. 缓存未命中 -> Agent 搜索（ReAct 模式）-> 存入缓存 -> 返回结果
        
        Args:
            task: 原始任务描述
            force_refresh: 是否强制刷新（跳过缓存）
            
        Returns:
            包含增强结果的字典：
            - enhanced_task: 增强后的任务描述
            - cache_hit: 是否命中缓存
            - search_performed: 是否执行了联网搜索
            - source: 结果来源 ("cache" / "agent" / "original")
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
        if not force_refresh:
            if self.verbose:
                print("\n[Step 1] 搜索向量缓存...")
            cache_result = await self.search_cache(task)
            
            if cache_result:
                if self.verbose:
                    print(f"命中缓存（相似度: {cache_result.get('similarity', 0):.4f}）")
                result["enhanced_task"] = cache_result.get("answer_text", task)
                result["cache_hit"] = True
                result["source"] = "cache"
                result["cache_similarity"] = cache_result.get("similarity", 0)
                return result
        else:
            if self.verbose:
                print("⚡ 强制刷新模式，跳过缓存")
        
        # Step 2: 缓存未命中，执行 Agent 搜索
        if self.enable_web_search:
            if self.verbose:
                print("\n[Step 2] 启动 Agent 搜索（ReAct 模式）...")
            
            # 在线程池中运行同步的 Agent
            enhanced_task, source_urls = await asyncio.to_thread(
                self._execute_agent_search, task
            )
            
            result["enhanced_task"] = enhanced_task
            result["search_performed"] = True
            result["source"] = "agent"
            result["source_urls"] = source_urls
        else:
            if self.verbose:
                print("\n⚠️ 联网搜索已禁用，使用原始任务")
        
        # Step 3: 将结果存入缓存
        if result["enhanced_task"] != task:
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