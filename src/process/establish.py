import json
from pathlib import Path


def convert_path_list_txt_to_a_jsonl(path_list_txt: Path, jsonl_path: Path):
    # Open the JSONL file for writing
    with open(jsonl_path, "w") as jsonl_file:
        # Open the text file containing paths
        with open(path_list_txt, "r") as f:
            for line in f:
                # Read each path and strip any extra whitespace
                json_path = line.strip()
                # Open and read the JSON file
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    # Write the JSON object as a line in the JSONL file
                    jsonl_file.write(json.dumps(json_data) + "\n")
