from action.action import Action
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any
from util.markdown_util import MarkdownUtil

SYSTEM_PROMPT = """
你是一个任务可行性分析助手。你需要判断给定的任务是否可以通过自动化测试框架执行。如果有历史任务，可以参考历史任务的执行情况。

请分析任务描述，判断：
1. 任务是否清晰明确
2. 任务是否可以通过 GUI 自动化工具执行
3. 任务是否有明确的执行步骤
"""

USER_PROMPT = """
# 历史任务情况
{history_tasks}

# 任务描述
{task}

请严格按照以下要求输出结果：
# 输出结构
使用 Markdown 标题分段
## 你的思考
你的思考过程

## 判断结果
只返回 "YES" 或 "NO"。

## 判断理由
判断的理由和依据
"""

class JudgmentTask(Action):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="judgment_task", description="判断测试任务是否可执行", llm=llm)

    async def run(self, history_tasks: str, task: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(history_tasks=history_tasks, task=task))
        ]
        response = await self.llm.ainvoke(messages)
        can_execute = True if "YES" in MarkdownUtil.extract_section(response.content, "判断结果") else False
        execution_reason = MarkdownUtil.extract_section(response.content, "判断理由")
        return can_execute, execution_reason






