import dataclasses
from builtins import dict
from src.make_data.framing import *
from typing import List, Any, Dict, Callable
from src.deploy.models import ModelChoice, TaskType


@dataclasses.dataclass
class Task:
    """Task dataclass"""

    name: str
    system: str
    shots: List[Dict[str, str]]
    task_type: TaskType
    model: ModelChoice
    default_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    return_json: bool = False
    deprecated: bool = False
    framing_func: Callable = None
    answer_framing_func: Callable = None
    acronym: str = ""
    json_fields: List[str] = dataclasses.field(default_factory=list)
    answer_fields: List[str] = dataclasses.field(default_factory=list)
    default_eval_func_str: str = ""
    is_eval_task: bool = False

    def dict(self):
        return {
            "name": self.name,
            "system": self.system,
            "shots": self.shots,
            "task_type": self.task_type,
            "default_params": self.default_params,
            "model": self.model,
            "return_json": self.return_json,
            "deprecated": self.deprecated,
            "framing_func": self.framing_func,
            "answer_framing_func": self.answer_framing_func,
            "acronym": self.acronym,
            "json_fields": self.json_fields,
            "answer_fields": self.answer_fields,
            "default_eval_func_str": self.default_eval_func_str,
            "is_eval_task": self.is_eval_task,
        }


@dataclasses.dataclass
class TaskChain:
    """TaskChain dataclass"""

    name: str
    task_list: List[str]
    post_processing_funcs: List[Callable]  # should be len (task_list) - 1

    def dict(self):
        return {
            "name": self.name,
            "task_list": self.task_list,
            "post_processing_funcs": self.post_processing_funcs,
        }


# A global registry for all tasks
available_tasks: Dict[str, Task] = {}

available_task_chains: Dict[str, List[str]] = {}


def register_task(task: Task, override: bool = False):
    """Register a task"""
    if not override:
        assert task.name not in available_tasks, f"Task {task.name} already exists"
    available_tasks[task.name] = task

    if not override:
        assert (
            task.acronym == "" or task.acronym not in available_tasks
        ), f"Task {task.acronym} already exists"
    available_tasks[task.acronym] = task


def register_task_chain(taskchain: TaskChain, override: bool = False):
    """Register a task"""
    if not override:
        assert (
            taskchain.name not in available_task_chains
        ), f"Task {taskchain.name} already exists"
    available_task_chains[taskchain.name] = taskchain




from .prompts import outline_revision_instruction

register_task(
    Task(
        name="outline-revision",
        system=outline_revision_instruction,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.7, frequency_penalty=0.9),
        model=ModelChoice.GPT_4_O,
        framing_func=framing4outline_revision,
        acronym="AB2C'",
        json_fields=["RevisedExcerpts", "SummaryOfChanges"],
        answer_fields=["edited_spec", "summary_of_change"],
        default_eval_func_str="gpt-score-outline-revision-v5",
    )
)




from .prompts import diff_analysis_instruction

register_task(
    Task(
        name="diff-analysis",
        system=diff_analysis_instruction,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.7, frequency_penalty=0.9),
        model=ModelChoice.GPT_4_O,
        framing_func=framing4diff_analysis,
        acronym="AC2B",
        default_eval_func_str="gpt-score-diff-analysis-v5",
    )
)



from .prompts import fill_cr_instruction

register_task(
    Task(
        name="fill-cr",
        system=fill_cr_instruction,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.7, frequency_penalty=0.9),
        model=ModelChoice.GPT_4_O,
        framing_func=framing4fill_cr,
        acronym="A+M2B",
        default_eval_func_str="gpt-score-fill-cr-v5",
    )
)



from .prompts import gpt_score_instruction_fill_cr_v4

register_task(
    Task(
        name="gpt-score-fill-cr-v5",
        system=gpt_score_instruction_fill_cr_v4,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.95),
        model=ModelChoice.GPT_4_O_MINI,
        framing_func=framing4gpt_score_fill_cr_v3,
        is_eval_task=True,
    )
)

from .prompts import gpt_score_instruction_outline_revision_v4

register_task(
    Task(
        name="gpt-score-outline-revision-v5",
        system=gpt_score_instruction_outline_revision_v4,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.95),
        model=ModelChoice.GPT_4_O_MINI,
        framing_func=framing4gpt_score_outline_revision_v3,
        is_eval_task=True,
    )
)


from .prompts import gpt_score_instruction_diff_analysis_v4

register_task(
    Task(
        name="gpt-score-diff-analysis-v5",
        system=gpt_score_instruction_diff_analysis_v4,
        task_type=TaskType.TEXT,
        shots=None,
        default_params=dict(temperature=0.2, top_p=0.95),
        model=ModelChoice.GPT_4_O_MINI,
        framing_func=framing4gpt_score_diff_analysis_v3,
        is_eval_task=True,
    )
)