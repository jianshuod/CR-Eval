import os
import time
import ftplib
import logging
import requests
import multiprocessing
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    filename="ftp_path_fetch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FTP server info
ftp_server = "ftp.3gpp.org"
ftp_user = "anonymous"
ftp_password = ""

failed_downloads = []


# Define a function to get the FTP path and download file
def get_ftp_path_and_download(wg_tdoc, output_dir):
    url = f"http://netovate.com/doc-search/?fname={wg_tdoc}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", class_="crtable")
    ftp_paths = []
    if table:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2 and wg_tdoc in cols[1].text:
                ftp_path = (
                    cols[1].find("a").get("href").replace("http://3gpp.org/ftp", "")
                )
                ftp_paths.append(ftp_path)
                # Download the file immediately after finding the path
                try:
                    download_file(ftp_path, output_dir)
                except Exception as e:
                    logger.error(f"Failed to download {ftp_path}: {e}")
                    time.sleep(3)  # Wait for 3 seconds before retrying
                    try:
                        download_file(ftp_path, output_dir)
                    except Exception as e:
                        logger.error(f"Retry failed to download {ftp_path}: {e}")
                        failed_downloads.append(ftp_path)
    return ",".join(ftp_paths)


# Define a function to handle retries
def get_ftp_path_with_retries(wg_tdoc, output_dir, retries=3):
    for attempt in range(retries):
        try:
            return get_ftp_path_and_download(wg_tdoc, output_dir)
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Error fetching {wg_tdoc}: {e}")
    return None


def download_file(remote_file_path, local_output_dir):
    """Downloads a file from the FTP server without progress display."""
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
    local_file_path = os.path.join(local_output_dir, os.path.basename(remote_file_path))

    resume_pos = 0
    file_exists = os.path.exists(local_file_path)

    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(user=ftp_user, passwd=ftp_password)

            total_size = ftp.size(remote_file_path)
            if total_size is None:
                total_size = 0

            if file_exists:
                local_file_size = os.path.getsize(local_file_path)
                if local_file_size == total_size:
                    logger.info(
                        f"File already exists and is complete: {local_file_path}"
                    )
                    return  # File is complete, skip download
                else:
                    os.remove(local_file_path)  # File is incomplete, delete it

            with open(
                local_file_path, "wb"
            ) as local_file:  # Open file in write mode to start fresh

                def callback(data):
                    local_file.write(data)

                ftp.retrbinary(f"RETR {remote_file_path}", callback)
                logger.info(f"Download completed: {local_file_path}")
    except ftplib.error_perm as e:
        logger.error(f"Permission error while downloading file: {e}")
    except Exception as e:
        logger.error(
            f"An error occurred while downloading file {remote_file_path}: {e}"
        )
        raise


# Define a wrapper function for the multiprocessing pool
def process_row(wg_tdoc_output_dir):
    wg_tdoc, output_dir = wg_tdoc_output_dir
    return get_ftp_path_with_retries(wg_tdoc, output_dir)


# Function to apply multiprocessing
def apply_multiprocessing(df, func, output_dir, num_processes):
    with multiprocessing.Pool(num_processes) as pool:
        tqdm.pandas()
        df["FTP Path"] = list(
            tqdm(
                pool.imap(func, [(wg_tdoc, output_dir) for wg_tdoc in df["WG Tdoc"]]),
                total=len(df),
            )
        )
    return df


def has_letters_and_sufficient_length(input_str):
    return (
        any(char.isalpha() for char in input_str)
        and len(input_str) >= 5
        and "-" in input_str
    )
