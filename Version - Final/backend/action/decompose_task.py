from subprocess import list2cmdline
from action.action import Action
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any
from util.markdown_util import MarkdownUtil

SYSTEM_PROMPT = """
你现在的角色是飞书桌面客户端自动化测试任务拆解助手。
请针对给定的“测试任务”，生成适合自动化执行的步骤列表，目标是指导一个 Windows 端 GUI 自动化框架来完成测试。如果有历史任务，你可以参考历史任务的拆解步骤。

请严格按照以下要求输出结果：

# 环境假设
1. 操作系统：Windows 10/11，中文环境。
2. 客户端：飞书桌面版已安装并能正常启动，并且已启动飞书客户端，已登录测试账户。

# 注意事项
1. 测试账户已提前登录飞书客户端，所有操作均在一个账户下实现。
2. 不能有模糊的概念和指代，不需要检查先前步骤是否正确，但需要确保每一步骤执行前后客户端都有明显的变化。
3. 似于‘复制粘贴’有较强逻辑顺序的操作需要放在一个步骤中。
4. 确保每个步骤都能独立执行。
5. 所有消息都发给‘测试账户’ 。
6. 步骤数量不能超过 5 步。

# 输出结构
使用 Markdown 标题分段
## 你的思考
你的思考过程

## 步骤列表
请以 JSON 格式返回步骤列表，格式如下：
{
  "steps": [
    {
      "id": "1",
      "step": "步骤描述",
    },
    {
      "id": "2",
      "step": "步骤描述",
    },
    {
      "id": "3",
      "step": "步骤描述",
    },
    ...
  ]
}
"""

USER_PROMPT = """
# 历史任务
{history_tasks}

# 测试任务
{task}
"""


class DecomposeTaskAction(Action):

    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="decompose_task", description="拆解测试任务", llm=llm)

    async def run(self, history_tasks: str, task: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(history_tasks=history_tasks, task=task))
        ]

        response = await self.llm.ainvoke(messages)
        response = MarkdownUtil.extract_json(response.content, "步骤列表")
        result  = response.get("steps", [])
        return result








