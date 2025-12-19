import os
import ftplib
import logging
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

# Configure logging
logging.basicConfig(
    filename="download_sr_cr_docs.log",
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


def extract_download_links(html_content):
    """Extract download links from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    links = soup.find_all("a", href=True)
    download_links = []
    for link in links:
        href = link["href"]
        parsed_url = urlparse(href)
        if parsed_url.netloc == "www.google.com" and "url" in parsed_url.path:
            query_params = parse_qs(parsed_url.query)
            if "q" in query_params:
                real_url = unquote(query_params["q"][0])
                if "3gpp.org" in real_url and real_url.endswith(".zip"):
                    download_links.append(real_url)
    return download_links


def convert_to_ftp_link(http_link):
    """Convert an HTTP link to an FTP link."""
    return http_link.replace("https://www.3gpp.org/ftp", "")


def download_from_html(html_file_path, output_dir):
    """Download all files from the HTML file."""
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    download_links = extract_download_links(html_content)
    ftp_links = [convert_to_ftp_link(link) for link in download_links]
    for ftp_link in ftp_links:
        download_file(ftp_link, output_dir)


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
