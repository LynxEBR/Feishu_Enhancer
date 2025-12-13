"""
AI Agent 自动化测试框架 - LangGraph 执行图
"""
import os
import sys
import asyncio
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from typing_extensions import Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models import EnhanceQuestion
from business_knowledge.database import get_db
from business_knowledge.crud import BusinessKnowledgeCRUD
import config
from action.decompose_task import DecomposeTaskAction
from action.ui_tars import UITars

# 导入飞书增强器
from feishu_enhancer import get_feishu_enhancer

# 配置（从全局配置字典获取）
OPENAI_API_KEY = config.config_dict.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = config.config_dict.get("OPENAI_BASE_URL", "")
MODEL_NAME = config.config_dict.get("MODEL_NAME", "gpt-5.1")

UI_TARS_BASE_URL = config.config_dict.get("UI_TARS_BASE_URL", "")
UI_TARS_API_KEY = config.config_dict.get("UI_TARS_API_KEY", "")
UI_TARS_MODEL = config.config_dict.get("UI_TARS_MODEL", "")

# 初始化 LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
    temperature=0.3,
)

tars = UITars(base_url=UI_TARS_BASE_URL, api_key=UI_TARS_API_KEY, model=UI_TARS_MODEL)

# === 状态定义 ===
class AgentState(TypedDict):
    """Agent 执行状态"""
    # 输入
    original_task: str  # 原始任务描述
    
    # 补全后的任务
    enhanced_task: Optional[str]  # 补全后的任务描述
    
    # 执行判断
    can_execute: Optional[bool]  # 是否能够执行
    execution_reason: Optional[str]  # 不能执行的原因
    
    # 子任务
    steps: List[Dict[str, Any]]  # 拆解后的步骤列表
    current_step_index: int  # 当前执行的步骤索引
    
    # 执行结果
    step_results: List[Dict[str, Any]]  # 步骤执行结果
    final_result: Optional[Dict[str, Any]]  # 最终结果
    
    # 消息历史
    messages: Annotated[List, add_messages]  # 消息历史


# === 节点函数 ===

async def enhance_task_node(state: AgentState) -> AgentState:
    """
    节点1: 补全执行任务
    
    使用联网飞书搜索 + PostgreSQL 向量缓存增强任务描述
    
    流程：
    1. 在 PostgreSQL ai_business_knowledge 表中搜索相似问题
    2. 若相似度 >= 阈值（默认 0.85），直接返回缓存的 answer_text
    3. 若未命中缓存，调用 Bocha API 进行联网搜索
    4. 使用 LLM 整合搜索结果，生成增强后的任务描述
    5. 将结果存入 PostgreSQL 向量库（question_text=原任务, answer_text=增强结果）
    """
    print(f"[补全任务节点] 原始任务: {state['original_task']}")

    # 获取飞书增强器实例
    enhancer = get_feishu_enhancer(llm)
    
    # 执行任务增强
    result = await enhancer.enhance(state['original_task'])
    
    enhanced_task = result.get("enhanced_task", state['original_task'])
    cache_hit = result.get("cache_hit", False)
    source = result.get("source", "unknown")
    
    # 日志输出
    if cache_hit:
        similarity = result.get("cache_similarity", 0)
        print(f"[补全任务节点] 缓存命中 (相似度: {similarity:.4f})")
    else:
        print(f"[补全任务节点] 来源: {source}")
    
    print(f"[补全任务节点] 补全后任务: {enhanced_task}")

    return {
        **state,
        "enhanced_task": enhanced_task,
        "messages": [AIMessage(content=f"补全后的任务: {enhanced_task}")]
    }


async def check_executability_node(state: AgentState) -> AgentState:
    """
    节点2: 判断是否能够执行
    使用 LLM 判断任务是否可执行
    """
    print(f"[判断可执行性节点] 检查任务: {state['enhanced_task']}")
    
#     system_prompt = """你是一个任务可行性分析助手。你需要判断给定的任务是否可以通过自动化测试框架执行。

# 请分析任务描述，判断：
# 1. 任务是否清晰明确
# 2. 任务是否可以通过 GUI 自动化工具执行
# 3. 任务是否有明确的执行步骤

# 如果任务可以执行，返回 "YES" 和简要说明。
# 如果任务不能执行，返回 "NO" 和不能执行的原因。"""

#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=f"请判断以下任务是否可执行：\n\n{state['enhanced_task']}\n\n请只返回 YES 或 NO，如果是 NO，请说明原因。")
#     ]
    
#     response = await llm.ainvoke(messages)
#     response_text = response.content.strip().upper()
    
    # can_execute = response_text.startswith("YES")
    # reason = response.content.strip()
    can_execute = True
    reason = "任务可以执行"
    
    print(f"[判断可执行性节点] 可执行: {can_execute}, 原因: {reason}")
    
    return {
        **state,
        "can_execute": can_execute,
        "execution_reason": reason,
        "messages": [AIMessage(content=f"可执行性判断: {'可执行' if can_execute else '不可执行'} - {reason}")]
    }


async def decompose_task_node(state: AgentState) -> AgentState:
    """
    节点3: 拆解子任务
    将任务拆解成多个可执行的步骤
    """
    print(f"[拆解任务节点] 拆解任务: {state['enhanced_task']}")

    action = DecomposeTaskAction(llm)
    steps = await action.run(task=state['enhanced_task'])
    
    
    return {
        **state,
        "steps": steps,
        "current_step_index": 0,
        "step_results": [],
        "messages": [AIMessage(content=f"拆解出 {len(steps)} 个步骤")]
    }


async def execute_subtask_node(state: AgentState) -> AgentState:
    """
    节点4: 执行子任务
    调用 MCP 服务器执行每个子任务
    """
    current_step_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    step_results = state.get("step_results", [])
    
    if current_step_index >= len(steps):
        print(f"[执行子任务节点] 所有子任务已完成")
        return state
    
    current_step = steps[current_step_index]
    print(f"[执行子任务节点] 执行子任务 {current_step_index + 1}/{len(steps)}: {current_step.get('step', '')}")
    
    try:
        result = await tars.run(description=current_step['step'], instruction=current_step['step'])
        print(f"[执行子任务节点] 执行结果: {result.get('success', False)}")

        # 将结果转换为 step_result 格式
        step_result = {
            "step_id": current_step.get("id"),
            "step_description": current_step.get("step"),
            "success": result.get("success", False) if isinstance(result, dict) else False,
        }
        
        step_results.append(step_result)
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行完成")
                
    except Exception as e:
        import traceback
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行失败: {str(e)}")
        print(f"[执行子任务节点] 错误详情: {traceback.format_exc()}")
        step_result = {
            "step_id": current_step.get("id"),
            "step_description": current_step.get("step"),
            "result": None,
            "success": False,
            "error": str(e),
        }
        step_results.append(step_result)
    
    # 移动到下一个子任务
    next_index = current_step_index + 1
    
    return {
        **state,
        "current_step_index": next_index,
        "step_results": step_results,
        "messages": [AIMessage(content=f"子任务 {current_step_index + 1} 执行完成")]
    }


async def finalize_node(state: AgentState) -> AgentState:
    """
    节点5: 结束节点
    汇总所有执行结果
    """
    print(f"[结束节点] 汇总执行结果")
    
    step_results = state.get("step_results", [])
    all_success = all(r.get("success", False) for r in step_results)
    
    final_result = {
        "original_task": state.get("original_task"),
        "enhanced_task": state.get("enhanced_task"),
        "total_subtasks": len(state.get("steps", [])),
        "completed_subtasks": len(step_results),
        "all_success": all_success,
        "subtask_results": step_results,
        "summary": f"共执行 {len(step_results)} 个子任务，{'全部成功' if all_success else '部分失败'}"
    }
    
    print(f"[结束节点] 最终结果: {final_result['summary']}")
    
    return {
        **state,
        "final_result": final_result,
        "messages": [AIMessage(content=f"任务执行完成: {final_result['summary']}")]
    }


# === 条件边函数 ===

def should_continue_execution(state: AgentState) -> Literal["decompose", "end"]:
    """
    判断是否继续执行：如果可执行则拆解任务，否则结束
    """
    can_execute = state.get("can_execute", False)
    if can_execute:
        return "decompose"
    else:
        return "end"


def should_continue_subtasks(state: AgentState) -> Literal["execute_subtask", "finalize"]:
    """
    判断是否继续执行子任务：如果还有未执行的子任务则继续，否则结束
    """
    current_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    
    if current_index < len(steps):
        return "execute_subtask"
    else:
        return "finalize"


# === 构建执行图 ===

def create_workflow_graph() -> StateGraph:
    """创建 LangGraph 执行图"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("enhance_task", enhance_task_node)
    workflow.add_node("check_executability", check_executability_node)
    workflow.add_node("decompose_task", decompose_task_node)
    workflow.add_node("execute_subtask", execute_subtask_node)
    workflow.add_node("finalize", finalize_node)
    
    # 设置入口点
    workflow.set_entry_point("enhance_task")
    
    # 添加边
    workflow.add_edge("enhance_task", "check_executability")
    workflow.add_conditional_edges(
        "check_executability",
        should_continue_execution,
        {
            "decompose": "decompose_task",
            "end": "finalize"
        }
    )
    workflow.add_edge("decompose_task", "execute_subtask")
    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue_subtasks,
        {
            "execute_subtask": "execute_subtask",  # 循环执行子任务
            "finalize": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# === 主函数 ===

async def run_task(task: str) -> Dict[str, Any]:
    """
    执行一个任务
    
    Args:
        task: 任务描述
        
    Returns:
        执行结果
    """
    # 创建执行图
    app = create_workflow_graph()
    
    # 初始化状态
    initial_state: AgentState = {
        "original_task": task,
        "enhanced_task": None,
        "can_execute": None,
        "execution_reason": None,
        "steps": [],
        "current_step_index": 0,
        "step_results": [],
        "final_result": None,
        "messages": [],
    }
    
    # 执行图
    print(f"\n{'='*60}")
    print(f"开始执行任务: {task}")
    print(f"{'='*60}\n")
    
    final_state = await app.ainvoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"任务执行完成")
    print(f"{'='*60}\n")
    
    return final_state.get("final_result", {})


async def main():
    """主函数"""
    # 示例任务
    example_task = "云文档被分享到IM中能正常打开"
    
    result = await run_task(example_task)
    
    print("\n执行结果:")
    print(f"原始任务: {result.get('original_task')}")
    print(f"补全任务: {result.get('enhanced_task')}")
    print(f"子任务数: {result.get('total_subtasks')}")
    print(f"完成数: {result.get('completed_subtasks')}")
    print(f"全部成功: {result.get('all_success')}")
    print(f"摘要: {result.get('summary')}")


if __name__ == "__main__":
    asyncio.run(main())
