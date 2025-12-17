import re
from typing import Optional, Any
import json

class MarkdownUtil:

    @staticmethod
    def extract_json(markdown_text: str, section_title: str) -> Optional[Any]:
        # 1. 找到标题所在位置（匹配任意级别的 #，例如 ## ### 都可以）
        heading_pattern = rf"^#+\s*{re.escape(section_title)}\s*$"
        heading_match = re.search(heading_pattern, markdown_text, flags=re.MULTILINE)
        if not heading_match:
            return None

        # 2. 截取该标题之后到下一个标题之前的内容
        start_pos = heading_match.end()
        remaining = markdown_text[start_pos:]

        next_heading_match = re.search(r"^#{1,6}\s+.+$", remaining, flags=re.MULTILINE)
        if next_heading_match:
            section_text = remaining[: next_heading_match.start()]
        else:
            section_text = remaining

        # 3. 优先从 ```json ... ``` 代码块中取出内容
        code_block_pattern = r"```(?:json)?\s*(.*?)```"
        code_block_match = re.search(code_block_pattern, section_text, flags=re.DOTALL | re.IGNORECASE)

        if code_block_match:
            json_candidate = code_block_match.group(1).strip()
        else:
            # 4. 如果没有 code block，就从小节文本中找第一个 { ... } 结构
            #    这里简单做法：从第一个 '{' 开始截取到最后一个 '}' 为止
            section_stripped = section_text.strip()
            first_brace = section_stripped.find("{")
            last_brace = section_stripped.rfind("}")
            if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
                return None
            json_candidate = section_stripped[first_brace : last_brace + 1].strip()

        # 5. 尝试解析 JSON
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            # 这里你可以改成日志打印
            print(f"[markdown_utils] JSON 解析失败: {e}\n原始文本:\n{json_candidate}")
            return None
        
    @staticmethod
    def extract_section(markdown_text: str, section_title: str) -> Optional[str]:
        """
        从 Markdown 文本中提取指定的二级标题 ## section_title 下的内容。
        
        Args:
            md_text (str): Markdown 文本
            section_title (str): 目标标题（不含 ##）
        
        Returns:
            Optional[str]: 标题下的内容，如果找不到则返回 None
        """
        # 构造标题正则（严格匹配 ## 结果）
        pattern = rf"^##\s+{re.escape(section_title)}\s*$"
        
        lines = markdown_text.splitlines()
        collecting = False
        collected = []

        for line in lines:
            if re.match(pattern, line.strip(), flags=re.IGNORECASE):
                # 找到目标标题，开始收集
                collecting = True
                continue

            # 遇到下一个 ## 终止
            if collecting and re.match(r"^##\s+", line.strip()):
                break

            if collecting:
                collected.append(line)

        if not collected:
            return None

        return "\n".join(collected).strip()