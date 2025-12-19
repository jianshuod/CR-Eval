import json
import dataclasses
from dataclasses import asdict
from typing import List, Tuple
from src.utils.logging import logging
from src.configs import MAX_CONTEXT_RATE
from src.analysis.utils import num_tokens_from_string
from src.process.labeling import labeling_reasoning_richness_single
from src.make_data.utils import (
    get_detailed_diff_list,
    is_json_serializable,
    get_masked_statements,
)

logger = logging.getLogger(__name__)


class IgnoreExtraFields:
    def __init__(self, *args, **kwargs):
        field_names = {f.name for f in dataclasses.fields(self)}
        for name in field_names:
            if name in kwargs:
                setattr(self, name, kwargs[name])
        # kwargs will ignore any fields not in the dataclass


@dataclasses.dataclass
class ChangeRequest(IgnoreExtraFields):
    spec: str = ""
    current_version: str = ""
    meeting_id: int = 0
    title: str = ""
    reason_for_change: str = ""
    summary_of_change: str = ""
    consequences_if_not_approved: str = ""
    clauses_affected: List[str] = dataclasses.field(default_factory=list)
    other_specs_affected: List[str] = dataclasses.field(default_factory=list)
    extracted_index: str = ""
    date: str = ""
    change_list: Tuple[str, str] = ("", "")
    table_modified_flag: bool = False
    page_count: int = 0
    figure_modified_flag: bool = False
    other_comments: str = ""
    source: str = ""
    source_to_wg: str = ""
    source_to_tsg: str = ""
    work_item_code: str = ""
    category: str = ""
    reasoning_richness: float = 0.0
    is_valid: bool = True
    invalid_reason: str = ""
    gold_response: str = ""
    sec_comment: str = ""
    sr_label: str = ""
    response_text: str = ""

    @property
    def diffRevision(self):
        revision_list, _ = get_detailed_diff_list(
            self.change_list[0], self.change_list[1]
        )

        diffRevision = "\n".join(
            [
                (
                    "\n".join([f"[-] {line}" for line in content])
                    if op == "[-]"
                    else (
                        "\n".join([f"[+] {line}" for line in modified])
                        if op == "[+]"
                        else (
                            "\n".join(
                                [f"[-] {line}" for line in content]
                                + [f"[+] {line}" for line in modified]
                            )
                            if op == "[>]"
                            else "\n".join([f"[ ] {line}" for line in content])
                        )
                    )
                )
                for op, content, modified in revision_list
            ]
        )

        return diffRevision

    @property
    def input_text_with_mask(self):
        return get_masked_statements(self.change_list[0], self.change_list[1])

    def __post_init__(self):
        self.reasoning_richness = labeling_reasoning_richness_single(self)

    def control_context_size(self, tokenizer, max_length):
        revision_list, _ = get_detailed_diff_list(
            self.change_list[0], self.change_list[1]
        )

        omitted_tag = "[X]"
        omitted_content = "==== Omitted ===="

        def calculate_estimated_context_ratio():
            context_statements = []
            revised_statements = []
            for op, original_content, _ in revision_list:
                if op in ("[ ]"):
                    context_statements.extend(original_content)
                if op in ("[-]", "[>]"):
                    revised_statements.extend(original_content)

            context_content = "\n".join(context_statements)
            revision_content = "\n".join(revised_statements)

            context_token_num = num_tokens_from_string(
                context_content, encoding=tokenizer
            )
            revision_token_num = num_tokens_from_string(
                revision_content, encoding=tokenizer
            )

            return context_token_num / (context_token_num + revision_token_num)

        estimated_context_ratio = calculate_estimated_context_ratio()

        while estimated_context_ratio > MAX_CONTEXT_RATE:
            removable_indices = [
                index for index, (op, _, _) in enumerate(revision_list) if op == "[ ]"
            ]

            if removable_indices:
                distances = []
                for idx in removable_indices:
                    distance = min(
                        (
                            abs(idx - other_idx)
                            for other_idx, (op, _, _) in enumerate(revision_list)
                            if op in ["[+]", "[-]", "[>]"]
                        ),
                        default=float("inf"),
                    )
                    distances.append((distance, idx))

                distances.sort()

                closest_idx = distances[-1][1]
                revision_list[closest_idx] = (
                    omitted_tag,
                    [omitted_content],
                    [omitted_content],
                )

                # Merge consecutive omitted tags after the loop
                i = 0
                while i < len(revision_list) - 1:
                    if (
                        revision_list[i][0] == omitted_tag
                        and revision_list[i + 1][0] == omitted_tag
                    ):
                        revision_list.pop(i + 1)
                    else:
                        i += 1
            else:
                # delete from the end if the context is still too long
                break

        self.change_list = (
            "\n".join(["\n".join(content) for op, content, _ in revision_list]),
            "\n".join(["\n".join(content) for op, _, content in revision_list]),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Ensure the key is a string
        if not isinstance(key, str):
            key = str(key)
        try:
            setattr(self, key, value)
        except AttributeError:
            pass


@dataclasses.dataclass
class RetrievalHit:
    """RetrievalHit dataclass"""

    id: str = ""
    text: str = ""
    score: float = 0.0

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclasses.dataclass
class EvaluationResult:
    """RetrievalHit dataclass"""

    eval_func_str: str = ""
    scoring_text: str = ""
    response_text: str = ""
    score: float = 0.0
    variance: float = 0.0

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)


@dataclasses.dataclass
class RetrievalInfo:
    """RetrievalInfo dataclass"""

    retrieval_mode: str = ""
    retrieval_topk: int = 0
    accepted_hits: List[RetrievalHit] = dataclasses.field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclasses.dataclass
class Instance:
    """Instance dataclass"""

    instance_id: str = ""
    cr_fields: ChangeRequest = dataclasses.field(default_factory=ChangeRequest)
    difficulty_level: int = 0
    task_name: str = ""
    eval_funcs_str: List[str] = dataclasses.field(default_factory=list)
    model_name: str = ""
    chunking_mode: str = ""
    max_context_length: int = 0
    document_encoding_func_str: str = ""
    retrieval_info: RetrievalInfo = dataclasses.field(default_factory=RetrievalInfo)
    input_text: str = ""
    time_cost: float = 0.0
    query_text: str = ""
    edited_spec: str = ""
    response_text: str = ""
    eval_results: List[EvaluationResult] = dataclasses.field(
        default_factory=EvaluationResult
    )
    sr_label: int = 0

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.cr_fields, key):
            return getattr(self.cr_fields, key)
        elif hasattr(self.retrieval_info, key):
            return getattr(self.retrieval_info, key)
        else:
            raise KeyError(f"{key} not found in Instance or ChangeRequest")

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self.cr_fields, key):
            setattr(self.cr_fields, key, value)
        elif hasattr(self.retrieval_info, key):
            setattr(self.retrieval_info, key, value)
        else:
            raise KeyError(f"{key} not found in Instance or ChangeRequest")

    def to_json(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, file_path: str) -> "Instance":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            data["cr_fields"] = ChangeRequest(**data["cr_fields"])
            data["retrieval_info"] = RetrievalInfo(
                accepted_hits=[
                    RetrievalHit(**hit)
                    for hit in data["retrieval_info"]["accepted_hits"]
                ],
                retrieval_mode=data["retrieval_info"]["retrieval_mode"],
                retrieval_topk=data["retrieval_info"]["retrieval_topk"],
            )
            return cls(**data)

    def to_jsonl(self, file_path=None) -> str:
        # Convert the dataclass to a dictionary
        data_dict = dataclasses.asdict(self)

        # Filter out non-serializable attributes
        serializable_dict = {
            k: v for k, v in data_dict.items() if is_json_serializable(v)
        }

        # Convert the dictionary to a JSON string without escaping non-ASCII characters
        json_str = json.dumps(serializable_dict, ensure_ascii=False)

        # If a file path is provided, append the JSON string to the file
        if file_path is not None:
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(json_str + "\n")

        # Return the JSON string representation of the instance
        return json_str

    @classmethod
    def from_jsonl(cls, file_path: str) -> List["Instance"]:
        instances = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                data["cr_fields"] = ChangeRequest(**data["cr_fields"])
                data["retrieval_info"] = RetrievalInfo(
                    accepted_hits=[
                        RetrievalHit(**hit)
                        for hit in data["retrieval_info"]["accepted_hits"]
                    ],
                    retrieval_mode=data["retrieval_info"]["retrieval_mode"],
                    retrieval_topk=data["retrieval_info"]["retrieval_topk"],
                )
                instances.append(cls(**data))
        return instances
