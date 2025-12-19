from functools import partial
from src.utils.logging import logging
from src.make_data.retrieval import RetrievalManager
from src.analysis.utils import num_tokens_from_string
from src.make_data.play_with_spec import build_spec_env
from src.make_data.instance import Instance, RetrievalInfo
from src.make_data.framing import apply_framing, concatenate_retrieved_text
from src.make_data.spec_processing.chunking import DOCUMENT_ENCODING_FUNCTIONS

logger = logging.getLogger(__name__)

logger.warning("Disabling caching")


def ingest_files(filenames):
    files_dict = dict()
    for filename in filenames:
        with open(filename) as f:
            content = f.read()
        files_dict[filename] = content
    return files_dict


def judge_difficulty_level(cr_fields):
    criterion2 = len(cr_fields["change_list"]) > 1
    criterion3 = len(cr_fields["other_specs_affected"]) != 0
    criterion4 = cr_fields["table_modified_flag"] or cr_fields["figure_modified_flag"]

    return sum([criterion2, criterion3, criterion4])


def setup_single_instance(
    cr_fields,
    max_context_length,
    output_dir,
    chunking_mode,
    document_encoding_func_str,
    task_name,
    model_name,
    retrieval_mode,
):

    document_encoding_func = DOCUMENT_ENCODING_FUNCTIONS[document_encoding_func_str]

    tokenizer_func = partial(num_tokens_from_string, model_choice=model_name)
    framing_func = partial(apply_framing, framing_func_str=task_name)

    instance_id = f"{cr_fields['spec']}-{cr_fields['extracted_index'].strip()}"
    difficulty_level = judge_difficulty_level(cr_fields)

    instance = Instance(
        instance_id=instance_id,
        cr_fields=cr_fields,
        difficulty_level=difficulty_level,
        max_context_length=max_context_length,
        chunking_mode=chunking_mode,
        task_name=task_name,
        model_name=model_name,
        document_encoding_func_str=document_encoding_func_str,
    )

    logger.info(
        f"Creating instance {instance_id}, the difficulty level of which is roughly {difficulty_level}"
    )

    if retrieval_mode == "oracle":

        instance["input_text"] = instance["change_list"][0]

    else:

        logger.info(f"Buidling retrieval index for {instance['instance_id']}")

        spec_context_manager = build_spec_env(
            cr_fields["spec"], cr_fields["current_version"], output_dir
        )

        retrieval_manager = RetrievalManager(
            spec_context_manager, chunking_mode, retrieval_mode
        )

        retrieval_manager.make_index(document_encoding_func, instance["instance_id"])

        hits = retrieval_manager.search(instance, "reason_for_change")
        logger.info(f"Retrieved {len(hits)} documents")

        base_text_input = framing_func(instance)
        base_text_input_length = len(tokenizer_func(base_text_input))

        accepted_hits = []
        cur_input_len = base_text_input_length

        for hit in hits:
            content = hit.text
            tokens = tokenizer_func(content)
            if cur_input_len + len(tokens) < max_context_length:
                accepted_hits.append(hit)
                cur_input_len += len(tokens)

        instance.retrieval_info = RetrievalInfo(
            retrieval_mode=retrieval_mode,
            retrieval_topk=len(accepted_hits),
            accepted_hits=accepted_hits,
        )

        instance["input_text"] = concatenate_retrieved_text(accepted_hits)

    instance["edited_spec"] = instance["change_list"][1]

    return instance
