import copy
import json
import os
from tqdm import tqdm  # Import tqdm for the progress bar
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
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

n = 10


def process_one_instance(jsonl_line, target_file):
    instance = json.loads(jsonl_line)

    if instance["task"] == "general":
        return

    messages = instance["messages"]
    orginal_response = messages[2]["content"]
    instance_to_be_revised = instance_to_be_revised_template.format(
        messages[0]["content"], messages[1]["content"], messages[2]["content"]
    )

    messages_to_query = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instance_to_be_revised},
    ]

    counter = 0
    while counter < n:
        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash-exp",
                messages=messages_to_query,
                temperature=0.8,
                top_p=0.95,
            )
            revised_response = response.choices[0].message.content
            new_instance = copy.deepcopy(instance)
            new_messages = copy.deepcopy(messages)
            new_messages[2]["content"] = revised_response

            new_instance["messages"] = new_messages
            new_instance["original-response"] = orginal_response

            target_file.write(json.dumps(new_instance) + "\n")
            target_file.flush()

            counter += 1
        except:
            continue


def process_batch(lines, batch_idx, batch_size, target_file):
    # Process a batch of lines using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_one_instance, line, target_file): idx
            for idx, line in enumerate(lines)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            # try:
            future.result()
        # except Exception as exc:
        #     print(f"Batch {batch_idx} Instance generated an exception: {exc}")


def process_instances_in_batches(lines, target_file, batch_size=100):
    for batch_idx in range(0, len(lines), batch_size):
        batch_lines = lines[batch_idx : batch_idx + batch_size]
        print(f"Processing batch {batch_idx // batch_size + 1}")
        process_batch(batch_lines, batch_idx // batch_size, batch_size, target_file)
