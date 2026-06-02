from abc import ABC, abstractmethod


class AgentSkill(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def tools(self) -> list: ...

    @property
    @abstractmethod
    def prompt_instructions(self) -> str: ...


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, AgentSkill] = {}

    def register(self, skill: AgentSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' já registrada")
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> AgentSkill | None:
        return self._skills.get(name)

    def all_tools(self) -> list:
        tools = []
        for s in self._skills.values():
            tools.extend(s.tools)
        return tools

    def all_prompt_instructions(self) -> str:
        return "\n\n".join(
            s.prompt_instructions
            for s in self._skills.values()
            if s.prompt_instructions
        )

    def list_skills(self) -> str:
        return "\n".join(
            f"- {s.name}: {s.description}" for s in self._skills.values()
        )


registry = SkillRegistry()
