import logging
import zipfile
from pathlib import Path


logging.basicConfig(
    filename="extract_cr_docs.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# logger = logging.get# logger(__name__)

failure_cases = []


def unzip_file(zip_path, extract_to):
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(zip_path):
        logging.error(f"{zip_path} is not a valid zip file.")
        return None

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        doc_files = [
            f for f in zip_ref.namelist() if f.endswith(".doc") or f.endswith(".docx")
        ]
        for file in doc_files:
            try:
                zip_ref.extract(file, extract_to)
                original_path = extract_to / file
                new_path = extract_to / Path(file).name
                original_path.rename(new_path)
                logging.info(f"Extracted {file} to {new_path}")
            except zipfile.BadZipFile as e:
                logging.error(f"Corrupt file {file} in {zip_path}: {e}")
                continue

        if len(doc_files) == 1:
            return extract_to / Path(doc_files[0]).name
        elif len(doc_files) > 1:
            logging.info("More than one .doc or .docx file extracted")
            return None
        else:
            logging.info("No .doc or .docx files found in the archive")
            return None


# Enhance error handling in process_cr_zip
def process_cr_zip(cr_path, target_dir):
    try:
        docx_path = unzip_file(cr_path, target_dir)
        if docx_path is None or not docx_path.suffix not in (".doc", ".docx"):
            logging.error(f"No valid .docx file found in {cr_path}")
            failure_cases.append(cr_path)
            return

    except Exception as e:
        logging.error(f"Failed to process {cr_path}: {str(e)}")
        failure_cases.append(cr_path)


def save_failure_cases(file_path):
    """Save the failure cases to a text file."""
    with open(file_path, "w") as file:
        for case in failure_cases:
            file.write(case + "\n")
