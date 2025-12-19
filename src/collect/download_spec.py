import io
import os
import re
import zipfile
import requests
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional, Union

FTP_LATEST_URL = "https://www.3gpp.org/ftp/Specs/latest"
FTP_ARCHIVE_URL = "https://www.3gpp.org/ftp/Specs/archive"


def download_spec(
    spec_id: str,
    container: str,
    output: Optional[Union[str, Path]] = None,
    spec_type: str = "latest",
    spec_version: Optional[str] = None,
) -> bool:
    """
    Downloads the latest or archived 3GPP specification with the given ID.

    Args:
        spec_id (str): The ID of the specification to download. Example: `24-301`
        container (str): The container to convert specification format
        output (str | Path, optional): The path to the output directory. Defaults to current working directory.
        spec_type (str, optional): The type of the specification. Either `latest` or `archive`. Defaults to `latest`.
        spec_version (str, optional): The version of the specification. Only used when spec_type is `archive`. Example: `19.0.0` or `j00`

    Returns:
        True if the download was successful.
    """

    def parse_spec_url(spec_id, spec_type="latest", spec_version=None):
        """
        Get the URL of the latest or archived 3GPP specification with the given ID.

        Args:
            spec_id (str): The ID of the specification to parse. Example: `24-301`
            spec_type (str, optional): The type of the specification. Either `latest` or `archive`.
            spec_version (str, optional): The version of the specification. Only used when spec_type is `archive`. Example: `19.0.0` or `j00`

        Returns:
            str: The URL of the latest or archived specification.

        Raises:
            ValueError: If the spec is not found.
        """

        if "-" in spec_id:
            spec_series = spec_id.split("-")[0]
            spec_index = spec_id.split("-")[1]
        elif "." in spec_id:
            spec_series = spec_id.split(".")[0]
            spec_index = spec_id.split(".")[1]
        else:
            spec_series = spec_id[0:2]
            spec_index = spec_id[2:5]

        if spec_type == "latest":
            spec_pattern = f"{spec_series}{spec_index}-...\\.zip"
            for i in range(18, 7, -1):
                series_dir = f"{FTP_LATEST_URL}/Rel-{i}/{spec_series}_series"

                # Get the list of all specs in the directory
                res = requests.get(series_dir)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, "html.parser")
                    spec_list: BeautifulSoup = soup.find("tbody").find_all("a")
                    spec_list = [(spec.text, spec["href"]) for spec in spec_list]
                    for spec in spec_list:
                        if re.match(spec_pattern, spec[0]):
                            return spec[1]
        else:
            if not spec_version:
                raise ValueError("Spec version is required for archive specs")

            if "." in spec_version:
                spec_version = "".join(
                    [
                        chr(ord("a") + int(x) - 10) if int(x) >= 10 else str(x)
                        for x in spec_version.split(".")
                    ]
                )

            spec_pattern = f"{spec_series}{spec_index}-{spec_version}\\.zip"
            series_dir = f"{FTP_ARCHIVE_URL}/{spec_series}_series"
            version_dir = f"{series_dir}/{spec_series}.{spec_index}"

            # Get the list of all specs in the directory
            res = requests.get(version_dir)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                spec_list: BeautifulSoup = soup.find("tbody").find_all("a")
                spec_list = [(spec.text, spec["href"]) for spec in spec_list]
                for spec in spec_list:
                    if re.match(spec_pattern, spec[0]):
                        return spec[1]

        # No spec found
        raise ValueError("Spec not found")

    def convert(src_file_path: str, container: str) -> None:
        """
        Convert .doc format to .docx

        Args:
            src_file_path (str): path to the source file
            container (str): container id

        Returns:
            None
        """

        dst_file_path = src_file_path + "x"
        file_name = src_file_path.split("/")[-1]
        with open(os.devnull, "w") as devnull:
            subprocess.run(
                ["docker", "cp", src_file_path, f"{container}:/root/"],
                stdout=devnull,
                stderr=devnull,
            )
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-it",
                    container,
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    "/root",
                    "/root/" + file_name,
                ],
                stdout=devnull,
                stderr=devnull,
            )
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container}:/root/" + file_name.replace(".doc", ".docx"),
                    dst_file_path,
                ],
                stdout=devnull,
                stderr=devnull,
            )

    output = Path(output) if output else Path.cwd()
    spec_url = parse_spec_url(spec_id, spec_type, spec_version)
    print(f"Downloading from {spec_url}, this may take a while...")

    res = requests.get(spec_url)
    if res.status_code != 200:
        raise ValueError("Failed to download spec")

    with zipfile.ZipFile(io.BytesIO(res.content)) as zip:
        for name in zip.namelist():
            if name.endswith(".docx") or name.endswith(".doc"):
                print(f"Downloaded {name} for spec {spec_id}. Unzipping...")
                zip.extract(name, output)

                if name.endswith(".doc"):
                    print(f"Converting to docx...")
                    convert((Path(output) / name).as_posix(), container)
                    (Path(output) / name).unlink()

            else:
                print(f"{name}: unknown file type, skipping...")
                continue

    return True
