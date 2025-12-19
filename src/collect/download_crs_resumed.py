import os
import ftplib
import logging
import requests
import multiprocessing
from tqdm.auto import tqdm
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    filename="ftp_path_fetch_resumed_0601.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FTP server info
ftp_server = "ftp.3gpp.org"
ftp_user = "anonymous"
ftp_password = ""

failed_downloads = []


def check_local_file_exists(remote_file_path, local_output_dir):
    """Check if the file already exists locally and has the correct size."""
    local_file_path = os.path.join(local_output_dir, os.path.basename(remote_file_path))
    if os.path.exists(local_file_path):
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(user=ftp_user, passwd=ftp_password)
            try:
                remote_size = ftp.size(remote_file_path)
                local_size = os.path.getsize(local_file_path)
                if local_size == remote_size:
                    return True
            except ftplib.error_perm:
                # File does not exist on the server
                pass
    return False


def download_file(remote_file_path, local_output_dir):
    """Download a file from the FTP server if it doesn't exist locally."""
    if not check_local_file_exists(remote_file_path, local_output_dir):
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)
        local_file_path = os.path.join(
            local_output_dir, os.path.basename(remote_file_path)
        )
        try:
            with ftplib.FTP(ftp_server) as ftp:
                ftp.login(user=ftp_user, passwd=ftp_password)
                with open(local_file_path, "wb") as local_file:
                    ftp.retrbinary(f"RETR {remote_file_path}", local_file.write)
                logger.info(f"Download completed: {local_file_path}")
        except ftplib.error_perm as e:
            logger.error(f"Permission error while downloading file: {e}")
            os.remove(local_file_path)
        except Exception as e:
            logger.error(
                f"An error occurred while downloading file {remote_file_path}: {e}"
            )
            os.remove(local_file_path)
            raise
    else:
        logger.info(f"Skipping download, file already exists: {remote_file_path}")


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
                download_file(ftp_path, output_dir)
    return ",".join(ftp_paths)


def process_row(wg_tdoc_output_dir):
    wg_tdoc, output_dir = wg_tdoc_output_dir
    return get_ftp_path_and_download(wg_tdoc, output_dir)


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
