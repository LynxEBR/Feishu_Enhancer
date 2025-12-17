from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from typing import Any

@dataclass
class Action(ABC):

    name: str
    description: str
    llm: ChatOpenAI

    def __init__(self, name: str = "", description: str = "", llm: ChatOpenAI = None):
        self.name = name
        self.description = description
        self.llm = llm

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

