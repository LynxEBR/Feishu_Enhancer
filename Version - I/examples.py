"""
飞书测试用例增强器 - 快速使用示例
"""

import asyncio
import os
from feishu_enhancer import FeishuTestCaseEnhancer


# ==================== 示例 1: 最简单的用法 ====================
def example_basic():
    """最基础的同步使用方式（脚本调用）"""
    print("\n" + "="*80)
    print("示例 1: 基础用法（同步）")
    print("="*80)
    
    # 初始化增强器
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=True
    )
    
    # 补全测试用例
    question = "云文档被分享到IM中能正常打开"
    result = enhancer.enhance(question)
    
    print(f"\n原始问题: {question}")
    print(f"补全结果: {result}")


# ==================== 示例 2: 异步方式 ====================
async def example_async():
    """异步使用方式（适合集成到 LangGraph）"""
    print("\n" + "="*80)
    print("示例 2: 异步方式")
    print("="*80)
    
    # 初始化增强器
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=True
    )
    
    # 异步补全测试用例
    question = "验证消息转发功能正常"
    result = await enhancer.aenhance(question)
    
    print(f"\n原始问题: {question}")
    print(f"补全结果: {result}")


# ==================== 示例 3: 测试缓存效果 ====================
async def example_cache():
    """展示缓存效果"""
    print("\n" + "="*80)
    print("示例 3: 缓存效果演示")
    print("="*80)
    
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=True
    )
    
    questions = [
        "云文档分享到IM",
        "验证云文档能分享到聊天窗口",  # 相似问题
        "测试文档分享到即时消息",      # 相似问题
    ]
    
    import time
    
    for i, q in enumerate(questions, 1):
        print(f"\n--- 第 {i} 次查询 ---")
        print(f"问题: {q}")
        
        start_time = time.time()
        result = await enhancer.aenhance(q)
        elapsed = time.time() - start_time
        
        print(f"耗时: {elapsed:.2f}秒")
        print(f"结果预览: {result[:80]}...")
    
    # 显示缓存统计
    stats = enhancer.get_cache_stats()
    print(f"\n缓存统计: {stats}")


# ==================== 示例 4: 批量处理 ====================
async def example_batch():
    """批量处理多个问题"""
    print("\n" + "="*80)
    print("示例 4: 批量处理")
    print("="*80)
    
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=False  # 批量处理时关闭详细日志
    )
    
    questions = [
        "测试消息撤回功能",
        "验证群聊@功能",
        "检查视频通话质量",
        "确认文件上传限制",
        "测试日历同步功能"
    ]
    
    print(f"开始批量处理 {len(questions)} 个问题...\n")
    
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] 处理: {q}")
        result = await enhancer.aenhance(q)
        results.append({
            'question': q,
            'enhanced': result[:100] + '...'  # 只显示前100字
        })
    
    print(f"\n 批量处理完成！")
    print(f"  缓存数量: {enhancer.get_cache_stats()['total_count']}")


# ==================== 示例 5: 集成到 main.py ====================
async def example_integration():
    """模拟 main.py 中的使用方式"""
    print("\n" + "="*80)
    print("示例 5: 集成到 main.py（模拟）")
    print("="*80)
    
    # 全局单例模式（推荐）
    global_enhancer = None
    
    def get_enhancer():
        global global_enhancer
        if global_enhancer is None:
            global_enhancer = FeishuTestCaseEnhancer(
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
                chroma_persist_dir="./chroma_db",
                similarity_threshold=0.90,
                verbose=True
            )
        return global_enhancer
    
    # 模拟 enhance_task_node
    async def enhance_task_node(original_task: str) -> str:
        """模拟 main.py 中的节点函数"""
        print(f"[节点] 原始任务: {original_task}")
        
        try:
            enhancer = get_enhancer()
            enhanced_task = await enhancer.aenhance(original_task)
            print(f"[节点] 补全后任务: {enhanced_task[:80]}...")
            return enhanced_task
        except Exception as e:
            print(f"[节点] 增强器失败: {e}")
            return original_task  # 回退到原始任务
    
    # 测试节点
    tasks = [
        "云文档分享测试",
        "消息转发验证",
    ]
    
    for task in tasks:
        enhanced = await enhance_task_node(task)
        print("-"*60)


# ==================== 示例 6: 错误处理 ====================
async def example_error_handling():
    """展示错误处理"""
    print("\n" + "="*80)
    print("示例 6: 错误处理")
    print("="*80)
    
    # 错误示例1: 缺少 API 密钥
    print("\n1. 测试缺少 API 密钥")
    try:
        enhancer = FeishuTestCaseEnhancer(
            dashscope_api_key=None  # 故意不提供
        )
    except ValueError as e:
        print(f"✓ 捕获到预期错误: {e}")
    
    # 正确的方式：提供密钥
    print("\n2. 正确初始化")
    enhancer = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        verbose=False
    )
    
    # 错误示例2: 网络问题（模拟）
    print("\n3. 测试带错误处理的调用")
    question = "测试网络异常场景"
    
    try:
        result = await enhancer.aenhance(question)
        print(f"✓ 成功: {result[:50]}...")
    except Exception as e:
        print(f"✗ 失败: {e}")
        print("建议: 检查网络连接和 API 密钥")


# ==================== 示例 7: 配置调优 ====================
def example_tuning():
    """展示不同配置的效果"""
    print("\n" + "="*80)
    print("示例 7: 配置调优")
    print("="*80)
    
    # 配置1: 高精度（严格匹配）
    print("\n配置1: 高精度模式")
    enhancer_strict = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        similarity_threshold=0.95,  # 高阈值
        verbose=False
    )
    print(f"  相似度阈值: 0.95")
    print(f"  特点: 只有非常相似的问题才命中缓存")
    
    # 配置2: 平衡模式（推荐）
    print("\n配置2: 平衡模式（推荐）")
    enhancer_balanced = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        similarity_threshold=0.90,  # 中等阈值
        verbose=False
    )
    print(f"  相似度阈值: 0.90")
    print(f"  特点: 兼顾精度和缓存命中率")
    
    # 配置3: 高命中率
    print("\n配置3: 高命中率模式")
    enhancer_loose = FeishuTestCaseEnhancer(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        similarity_threshold=0.85,  # 低阈值
        verbose=False
    )
    print(f"  相似度阈值: 0.85")
    print(f"  特点: 更容易命中缓存，但可能不太精确")


# ==================== 主函数 ====================
async def main():
    """运行所有示例"""
    print("\n" + "🎬"*30)
    print("飞书测试用例增强器 - 使用示例集")
    print("🎬"*30)
    
    # 检查 API 密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请执行: export DASHSCOPE_API_KEY='your_key'")
        return
    
    # 示例菜单
    examples = [
        ("基础用法（同步）", example_basic, False),
        ("异步方式", example_async, True),
        ("缓存效果演示", example_cache, True),
        ("批量处理", example_batch, True),
        ("集成到 main.py", example_integration, True),
        ("错误处理", example_error_handling, True),
        ("配置调优", example_tuning, False),
    ]
    
    print("\n可用示例：")
    for i, (name, _, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 运行所有示例")
    
    choice = input("\n请选择示例编号 (0-7): ").strip()
    
    if choice == "0":
        # 运行所有示例
        for name, func, is_async in examples:
            print(f"\n{'='*80}")
            print(f"运行示例: {name}")
            print(f"{'='*80}")
            
            if is_async:
                await func()
            else:
                func()
            
            input("\n按回车继续...")
    
    elif choice in [str(i) for i in range(1, len(examples) + 1)]:
        # 运行单个示例
        idx = int(choice) - 1
        name, func, is_async = examples[idx]
        
        if is_async:
            await func()
        else:
            func()
    else:
        print("无效选择")
    
    print("\n" + "✨"*30)
    print("示例结束！")
    print("✨"*30)


if __name__ == "__main__":
    asyncio.run(main())
