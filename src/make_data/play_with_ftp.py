import os
import time
import ftplib
import zipfile
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
from src.utils.logging import logging

# FTP server info
ftp_server = "ftp.3gpp.org"
ftp_user = "anonymous"
ftp_password = ""

ftp = ftplib.FTP(ftp_server)
ftp.login(user=ftp_user, passwd=ftp_password)

logger = logging.getLogger(__name__)


def download_file(remote_file_path, local_output_dir):
    """Downloads a file from the FTP server with progress display."""
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
    local_file_path = os.path.join(local_output_dir, os.path.basename(remote_file_path))
    logger.info(f"Downloading file: {remote_file_path} to {local_file_path}")

    downloaded_size = 0
    start_time = time.time()
    resume_pos = 0

    if os.path.exists(local_file_path):
        resume_pos = os.path.getsize(local_file_path)

    with open(local_file_path, "ab") as local_file:
        try:
            if resume_pos > 0:
                ftp.sendcmd(f"REST {resume_pos}")
            # know about the size of remote file
            total_size = ftp.size(remote_file_path)
            if total_size is None:
                total_size = 0

            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as pbar:

                def callback(data):
                    local_file.write(data)
                    nonlocal downloaded_size
                    downloaded_size += len(data)
                    pbar.update(len(data))

                ftp.retrbinary(f"RETR {remote_file_path}", callback)
        except ftplib.error_perm as e:
            logging.error(f"\nPermission error while downloading file: {e}")
        except Exception as e:
            logging.error(
                f"\nAn error occurred while downloading file {remote_file_path}: {e}"
            )
            raise


def download_cr(cr_index: str, output_dir: str):
    """Downloads the CR document from the FTP server."""
    local_file_path = os.path.join(output_dir, os.path.basename(cr_index))
    remote_cr_path = "/ftp/Specs/archive/" + cr_index
    download_file(remote_cr_path, local_file_path)


def convert_version_to_abc_form(version: str):
    mapping = {i: i for i in range(0, 10)}
    additional_mapping = {
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        16: "g",
        17: "h",
        18: "i",
        19: "j",
    }
    mapping.update(additional_mapping)
    return "".join([str(mapping[int(v)]) for v in version.split(".")])


def unzip_file(zip_file_path, extract_to_dir):
    """Unzips the file and counts the number of extracted files."""
    logger.info(f"Unzipping file: {zip_file_path} to {extract_to_dir}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)
        extracted_files = zip_ref.namelist()

    file_count = len(extracted_files)
    logger.info(f"Extracted {file_count} files.")
    return file_count


def convert_doc_to_docx(doc_path, out_dir, container_name="libreoffice"):
    doc_path = Path(doc_path)
    out_dir = Path(out_dir)
    docx_path = out_dir / doc_path.with_suffix(".docx").name

    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define paths in the container
    container_input_dir = "/input"
    container_output_dir = "/output"
    container_doc_path = Path(container_input_dir) / doc_path.name
    container_docx_path = (
        Path(container_output_dir) / doc_path.with_suffix(".docx").name
    )

    try:
        # Run the conversion command inside the container
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "libreoffice",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                container_output_dir,
                str(container_doc_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"Converted {doc_path} to {docx_path}")
        logger.debug(f"LibreOffice output: {result.stdout}")

        # Remove the files in the container
        subprocess.run(
            ["docker", "exec", container_name, "rm", "-f", str(container_doc_path)],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        logger.warning(f"Error during conversion: {e}")
        logger.warning(f"Stderr: {e.stderr}")
        logger.warning(f"Stdout: {e.stdout}")
        return None

    return docx_path


def download_spec(spec_index: str, version: str, output_dir: str):
    """Downloads the specification document from the FTP server.

    - outpur_dir
        - local_output_dir
            - .zip
            - -f.docx (a ready version of the spec file)

    """

    spec_id = f"{spec_index.replace('.', '')}-{convert_version_to_abc_form(version)}"

    local_output_dir = os.path.join(output_dir, spec_id)
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    local_file_path = os.path.join(local_output_dir, f"{spec_id}--f.docx")
    if not os.path.exists(local_file_path):
        # Automatically fetch the spec from the ftp server
        remote_spec_path = f'/Specs/archive/{spec_index.split(".")[0]}_series/{spec_index}/{spec_id}.zip'
        download_file(remote_spec_path, local_output_dir)
        file_count = unzip_file(
            os.path.join(local_output_dir, os.path.basename(remote_spec_path)),
            local_output_dir,
        )
        if file_count == 1:
            for file_name in os.listdir(local_output_dir):
                file_name = os.path.join(local_output_dir, file_name)
                if file_name.find(".doc") == -1:
                    continue

                if file_name.find(".docx") == -1:
                    convert_doc_to_docx(file_name, local_output_dir)

                os.rename(local_file_path.replace("--f", ""), local_file_path)
        else:  # TODO: handle the case when the zip file contains multiple files
            pass

    return local_file_path
