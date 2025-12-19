import json
import copy
from tqdm import tqdm  # Import tqdm for the progress bar
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="token-abc123",
)


prompt = """You will be given a task instance composed of TASK INSTRUCTION, USER QUERY, and ASSISTANT RESPONSE. Your task is to revise the ASSISTANT RESPONSE by adding reasoning contents to it. The reasoning contents should explain how the response was generated and act as chain of thoughts for reaching the responses.

Note that
- The task will be related to network protocols, and you should leverage your knowledge in this domain.
- The revised response should be coherent with the original response.
- The revised response should perfectly fit the TASK INSTRUCTION and USER QUERY.
- The revised response should be informative and helpful to the user.
- The revised response should be rich in thoughts and smooth in logic.
- The revised response should be fruitful in educating other assistants.
- You should not alter the original response format.
- You should only return the revised response, which can directly replace the original response."""

instance_to_be_revised_template = """# TASK INSTRUCTION

{}

# USER QUERY

{}

# ASSISTANT RESPONSE

{}"""

n = 10


def process_one_instance(jsonl_line):
    instance = json.loads(jsonl_line)

    if instance["task"] == "general":
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
    # Process a batch of lines using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(process_one_instance, line): idx
            for idx, line in enumerate(lines)
        }
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Batch {batch_idx} Instance generated an exception: {exc}")
    return results


def process_instances_in_batches(lines, batch_size=640):
    counter = 0
    for batch_idx in range(0, len(lines), batch_size):
        counter += 1
        if counter <= 3:
            continue
        batch_lines = lines[batch_idx : batch_idx + batch_size]
        print(f"Processing batch {batch_idx // batch_size + 1}")
        results = process_batch(batch_lines, batch_idx // batch_size, batch_size)
        yield batch_idx, results
