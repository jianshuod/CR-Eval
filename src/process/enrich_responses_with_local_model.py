import time
import copy
import json
from tqdm import tqdm  # Import tqdm for the progress bar
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="token-abc123",
)

prompt = """You will be given a task instance composed of TASK INSTRUCTION, USER QUERY, and ASSISTANT RESPONSE. Your task is to revise the ASSISTANT RESPONSE by adding reasoning contents to it. The reasoning contents should explain how the response was generated and act as chain of thoughts for reaching the responses.

Note that
- You are highly encouraged to utilize your understandings and knowledge about the task when enriching the response.
- The task will be related to **cellular network protocols** and **SECURITY**, and you should leverage your knowledge in this domain.
- The revised response should be coherent with the original response.
- The revised response should perfectly fit the TASK INSTRUCTION and USER QUERY.
- The revised response should be informative and helpful to the user.
- The revised response should be rich in thoughts and smooth in logic.
- The revised response should be fruitful in educating other assistants, especially in fostering security experts of cellular network protocols.
- You should not alter the original response format.
- You should only return the revised response strictly adhering to the original response format with no additional headers, which can directly replace the original response."""

instance_to_be_revised_template = """# TASK INSTRUCTION

{}

# USER QUERY

{}

# ASSISTANT RESPONSE

{}"""

n = 5


def process_one_instance_with_retry(jsonl_line, max_retries=2, retry_delay=120):
    """
    Processes one JSONL instance with retry and timeout logic.
    """
    retries = 0
    while retries <= max_retries:
        try:
            return process_one_instance(jsonl_line)  # Attempt to process the instance
        except Exception as exc:
            retries += 1
            if retries > max_retries:
                print(f"Max retries exceeded for instance. Error: {exc}")
                return []  # Return an empty result if retries are exhausted
            # print(f"Retrying instance due to error: {exc}. Attempt {retries} of {max_retries}")
            time.sleep(retry_delay)  # Wait before retrying


def process_one_instance(jsonl_line):
    """
    Processes a single JSONL line.
    """
    instance = json.loads(jsonl_line)

    if instance["task"] in ("general", "outline-revision"):
        return []

    messages = instance["messages"]
    orginal_response = messages[2]["content"]
    instance_to_be_revised = instance_to_be_revised_template.format(
        messages[0]["content"], messages[1]["content"], messages[2]["content"]
    )

    messages_to_query = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instance_to_be_revised},
    ]

    response = client.chat.completions.create(
        model="Meta-Llama-3.1-70B-Instruct",
        messages=messages_to_query,
        temperature=0.8,
        top_p=0.95,
        n=n,
    )

    enriched_instances = []

    for i in range(n):
        revised_response = response.choices[i].message.content
        new_instance = copy.deepcopy(instance)
        new_messages = copy.deepcopy(messages)
        new_messages[2]["content"] = revised_response

        new_instance["messages"] = new_messages
        new_instance["original-response"] = orginal_response
        enriched_instances.append(json.dumps(new_instance))

    return enriched_instances


def process_batch(lines, batch_idx, batch_size):
    """
    Processes a batch of lines using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(process_one_instance_with_retry, line): idx
            for idx, line in enumerate(lines)
        }
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Batch {batch_idx} Instance generated an exception: {exc}")
    return results


def process_instances_in_batches(lines, batch_size=3200):
    """
    Processes instances in batches.
    """
    for batch_idx in range(0, len(lines), batch_size):
        if batch_idx // batch_size < 858:
            continue
        batch_lines = lines[batch_idx : batch_idx + batch_size]
        print(f"Processing batch {batch_idx // batch_size + 1}")
        results = process_batch(batch_lines, batch_idx // batch_size, batch_size)
        yield batch_idx, results
