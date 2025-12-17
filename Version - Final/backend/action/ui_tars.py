"""
UI-TARS Desktop Action
将 UI-TARS 客户端功能直接集成到 Action 中，用于执行桌面自动化任务
"""
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple
from action.action import Action
from langchain_openai import ChatOpenAI


class UITars(Action):
    """UI-TARS Desktop Action，用于执行自然语言桌面自动化指令"""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_responses_api: bool = False,
        ui_tars_cli_path: Optional[str] = None,
    ):
        """
        初始化 UI-TARS
        
        Args:
            llm: LLM 实例（可选，UI-TARS Action 不需要 LLM）
            base_url: VLM 模型的基础 URL
            api_key: API 密钥
            model: 模型名称
            use_responses_api: 是否使用 Responses API
            ui_tars_cli_path: UI-TARS CLI 的路径（如果已安装）
        """
        super().__init__(
            name="ui_tars",
            description="执行桌面自动化任务，通过自然语言控制桌面程序",
            llm=llm
        )
        
        # 从配置或参数获取模型配置
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.use_responses_api = use_responses_api
        
        # CLI 路径配置
        self.ui_tars_cli_path = ui_tars_cli_path
        self.config_file_path = Path.home() / ".ui-tars-cli.json"
        
        # 尝试查找本地 UI-TARS-desktop 项目的 CLI
        self.local_cli_path = self._find_local_cli()
        
        # 确保配置完整
        if not all([self.base_url, self.api_key, self.model]):
            raise ValueError(
                "缺少必要的配置信息。请设置 UI_TARS_BASE_URL、UI_TARS_API_KEY 和 UI_TARS_MODEL"
            )
        
        # 保存配置到 CLI 配置文件
        self._save_cli_config()
    
    def _save_cli_config(self):
        """保存配置到 UI-TARS CLI 配置文件"""
        config = {
            "baseURL": self.base_url,
            "apiKey": self.api_key,
            "model": self.model,
            "useResponsesApi": self.use_responses_api
        }
        
        try:
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告: 无法保存 CLI 配置文件: {e}")
    
    def _find_npx_command(self) -> Optional[str]:
        """查找可用的 npx 命令"""
        if sys.platform == "win32":
            for cmd_name in ["npx.cmd", "npx"]:
                try:
                    result = subprocess.run(
                        ["where", cmd_name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return cmd_name
                except:
                    continue
        else:
            try:
                result = subprocess.run(
                    ["which", "npx"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return "npx"
            except:
                pass
        return None
    
    
    def _find_local_cli(self) -> Optional[str]:
        """查找本地 UI-TARS-desktop 项目的 CLI"""
        # 从当前文件位置向上查找 UI-TARS-desktop 目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # action -> backend -> project root
        
        # 可能的 UI-TARS-desktop 路径
        possible_paths = [
            project_root / "UI-TARS-desktop" / "packages" / "ui-tars" / "cli" / "bin" / "index.js",
            project_root.parent / "UI-TARS-desktop" / "packages" / "ui-tars" / "cli" / "bin" / "index.js",
            Path("UI-TARS-desktop") / "packages" / "ui-tars" / "cli" / "bin" / "index.js",
        ]
        
        for cli_path in possible_paths:
            if cli_path.exists():
                return str(cli_path)
        
        return None
    
    def _execute_command(
        self,
        instruction: str,
        target: str = "nut-js",
        timeout: Optional[int] = None,
        verbose: bool = False
    ) -> dict[str, Any]:
        """
        执行自然语言指令（内部方法）
        
        Args:
            instruction: 自然语言指令，例如 "打开微信"、"点击开始菜单" 等
            target: 目标操作器类型，可选值: "nut-js" (桌面), "adb" (Android)
            timeout: 超时时间（秒），None 表示不设置超时
            verbose: 是否显示详细输出
        
        Returns:
            包含执行结果的字典
        """
        if not instruction or not instruction.strip():
            raise ValueError("指令不能为空")
        
        # 构建命令
        # 优先级：1. 用户指定的路径 2. 本地项目 CLI 3. 全局 CLI 4. npx
        if self.ui_tars_cli_path:
            cmd = [self.ui_tars_cli_path, "start"]
        elif self.local_cli_path:
            cmd = ["node", self.local_cli_path, "start"]
        else:
            # 检查全局 CLI 或使用 npx
            global_cli_installed, cli_path = self._check_global_cli()
            if global_cli_installed:
                if cli_path:
                    cmd = ["node", cli_path, "start"] if Path(cli_path).suffix.lower() == '.js' else [cli_path, "start"]
                else:
                    cmd = ["cmd", "/c", "ui-tars", "start"] if sys.platform == "win32" else ["ui-tars", "start"]
            else:
                npx_cmd = self._find_npx_command() or "npx"
                if sys.platform == "win32":
                    cmd = ["cmd", "/c", npx_cmd, "@ui-tars/cli@latest", "start"]
                else:
                    cmd = [npx_cmd, "@ui-tars/cli@latest", "start"]
        
        # 添加参数
        cmd.extend([
            "-t", target,
            "-q", instruction.strip()
        ])
        
        if verbose:
            print(f"执行命令: {' '.join(cmd)}")
            print(f"指令: {instruction}")
            if self.local_cli_path:
                print(f"使用本地 CLI: {self.local_cli_path}")
            elif self.ui_tars_cli_path:
                print(f"使用指定 CLI: {self.ui_tars_cli_path}")
            else:
                print("使用全局 CLI 或 npx")
        
        try:
            # 设置执行参数
            shell = sys.platform == "win32" and cmd[0] == "cmd"
            cwd = None
            
            # 如果使用本地 CLI，设置工作目录为 UI-TARS-desktop 根目录
            if self.local_cli_path and cmd[0] == "node" and self.local_cli_path in cmd[1]:
                ui_tars_root = Path(self.local_cli_path).parent.parent.parent.parent.parent
                if ui_tars_root.exists() and (ui_tars_root / "package.json").exists():
                    cwd = str(ui_tars_root)
                    if verbose:
                        print(f"工作目录: {cwd}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                shell=shell,
                cwd=cwd
            )
            
            # 返回结果
            return {
                "success": result.returncode == 0,
                "instruction": instruction,
                "target": target,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.stderr if result.returncode != 0 else None
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "instruction": instruction,
                "target": target,
                "error": f"执行超时（{timeout}秒）"
            }
        
        except FileNotFoundError as e:
            error_detail = str(e)
            return {
                "success": False,
                "instruction": instruction,
                "target": target,
                "error": f"找不到命令: {error_detail}"
            }
        
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是模块缺失错误
            if "Cannot find module" in error_msg or "MODULE_NOT_FOUND" in error_msg:
                error_msg = error_msg
            
            return {
                "success": False,
                "instruction": instruction,
                "target": target,
                "error": error_msg
            }
    
    async def run(
        self,
        description: str = "",
        query: str = "",
        instruction: Optional[str] = None,
        target: str = "nut-js",
        timeout: Optional[int] = None,
        verbose: bool = False
    ) -> dict[str, Any]:
        """
        执行自然语言指令
        
        Args:
            description: 步骤描述（可选）
            query: 执行指令（主要参数，例如 "打开微信"、"点击开始菜单" 等）
            instruction: 自然语言指令（如果提供，将优先使用此参数）
            target: 目标操作器类型，可选值: "nut-js" (桌面), "adb" (Android)
            timeout: 超时时间（秒），None 表示不设置超时
            verbose: 是否显示详细输出
            
        Returns:
            包含执行结果的字典，格式：
            {
                "success": bool,
                "instruction": str,
                "target": str,
                "returncode": int,
                "stdout": str,
                "stderr": str,
                "error": Optional[str]
            }
        """
        # 优先使用 instruction 参数，否则使用 query 参数
        instruction_text = instruction or query or description
        
        if not instruction_text or not instruction_text.strip():
            raise ValueError("指令不能为空，请提供 query 或 instruction 参数")
        
        # 使用 asyncio.to_thread 将同步的 subprocess.run 转换为异步执行
        # 这样可以避免阻塞事件循环，确保顺序执行
        import asyncio
        result = await asyncio.to_thread(
            self._execute_command,
            instruction=instruction_text.strip(),
            target=target,
            timeout=timeout,
            verbose=verbose
        )
        
        return result
