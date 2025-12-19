import re
import zipfile
import subprocess
from pathlib import Path
from docx.oxml.ns import qn, nsmap
from src.utils.logging import logging

logger = logging.getLogger(__name__)


def convert_doc_to_docx(doc_path):
    doc_path = Path(doc_path)

    if doc_path.suffix == ".docx":
        return doc_path

    docx_path = doc_path.with_suffix(".docx")

    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(doc_path.parent),
                str(doc_path),
            ],
            check=True,
        )
        # logger.info(f"Converted {doc_path} to {docx_path}")
    except subprocess.CalledProcessError as e:
        # logger.warning(f'Error during conversion: {e}')
        return docx_path


def detect_table_modification(doc):
    if len(doc.tables) > 3:

        for table in doc.tables[3:]:
            if len(table.rows) > 1:
                return True

    return False


def count_images(doc):
    image_count = 0

    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_count += 1

    return image_count


def getDocxPageCount(docx_fpath):
    # logger.info(f"Getting page count of {docx_fpath}")
    try:
        docx_object = zipfile.ZipFile(docx_fpath)
        docx_property_file_data = docx_object.read("docProps/app.xml").decode()
        match = re.search(r"<Pages>(\d+)</Pages>", docx_property_file_data)
        if match:
            page_count = match.group(1)
            return int(page_count)
        else:
            # Log an error or print if no pages tag is found
            # logger.error("No <Pages> tag found in docProps/app.xml")
            return None
    except zipfile.BadZipFile:
        logger.error(f"File {docx_fpath} is not a valid zip file or is corrupted.")
        return None
    except FileNotFoundError:
        logger.error(f"File {docx_fpath} not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None


def get_cell_text(cell):
    def _get_cell_text(element):
        text = ""
        for child in element:
            if child.tag == qn("w:t"):
                text += child.text or ""
            else:
                text += _get_cell_text(child)
        return text

    nsmap["mc"] = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    return _get_cell_text(cell._element)


def print_xml_structure(element, indent=0):
    spaces = " " * indent
    print(f"{spaces}{element.tag}: {element.text}")

    for child in element:
        print_xml_structure(child, indent + 2)


def deny_all_text(p):

    def _deny_all(p):
        text = ""
        for run in p.xpath(
            "w:r[not(ancestor::w:ins)] | w:del | w:del/w:r | w:moveFrom/w:r"
        ):
            for child in run:
                if child.tag == qn("w:t"):
                    text += child.text or ""
                elif child.tag == qn("w:tab"):
                    text += "\t"
                elif child.tag in (qn("w:br"), qn("w:cr")):
                    text += "\n"
                elif child.tag == qn("w:delText"):
                    text += child.text or ""
                elif child.tag == qn("w:del"):
                    for del_run in child.xpath("w:r"):
                        for del_child in del_run:
                            if del_child.tag == qn("w:t"):
                                text += del_child.text or ""
                            elif del_child.tag == qn("w:tab"):
                                text += "\t"
                            elif del_child.tag in (qn("w:br"), qn("w:cr")):
                                text += "\n"
                            elif del_child.tag == qn("w:delText"):
                                text += del_child.text or ""
                elif child.tag == qn("mc:AlternateContent"):
                    for nested_p in child.xpath("mc:Choice[1]//w:p", namespaces=nsmap):
                        text += _deny_all(nested_p)
                        text += "\n"
        return text

    nsmap["mc"] = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    nsmap["v"] = "urn:schemas-microsoft-com:vml"
    nsmap["o"] = "urn:schemas-microsoft-com:office:office"
    nsmap["a"] = "http://schemas.openxmlformats.org/drawingml/2006/main"

    return _deny_all(p._p)


def accept_all_text(p):

    def _accepted_text(p):
        text = ""
        for run in p.xpath("w:r[not(w:pPr/w:rPr/w:moveFrom)] | w:ins/w:r"):
            for child in run:
                if child.tag == qn("w:t"):
                    text += child.text or ""
                elif child.tag == qn("w:tab"):
                    text += "\t"
                elif child.tag in (qn("w:br"), qn("w:cr")):
                    text += "\n"
                elif child.tag == qn("mc:AlternateContent"):
                    for nested_p in child.xpath("mc:Choice[1]//w:p", namespaces=nsmap):
                        text += _accepted_text(nested_p)
                        text += "\n"
        return text

    nsmap["mc"] = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    nsmap["v"] = "urn:schemas-microsoft-com:vml"
    nsmap["o"] = "urn:schemas-microsoft-com:office:office"
    nsmap["a"] = "http://schemas.openxmlformats.org/drawingml/2006/main"

    return _accepted_text(p._p)


from docx.oxml.ns import qn


def deny_all_table(table):
    def _deny_all_text(p):
        text = ""
        for run in p.iter(qn("w:r")):
            if run.find(qn("w:ins")) is None:
                for child in run:
                    if child.tag == qn("w:t"):
                        text += child.text or ""
                    elif child.tag == qn("w:tab"):
                        text += "\t"
                    elif child.tag in (qn("w:br"), qn("w:cr")):
                        text += "\n"
                    elif child.tag == qn("w:delText"):
                        text += child.text or ""
                    elif child.tag == qn("mc:AlternateContent"):
                        for nested_p in child.iter(qn("mc:Choice")):
                            text += _deny_all_text(nested_p)
                            text += "\n"
        return text

    table_text = ""
    for row in table._element.iter(qn("w:tr")):
        for cell in row.iter(qn("w:tc")):
            for p in cell.iter(qn("w:p")):
                if p.find(qn("w:ins")) is None:
                    table_text += _deny_all_text(p)
                    table_text += "\t"
        table_text = table_text.rstrip("\t")
        table_text += "\n"
    return table_text.rstrip("\n")


def accept_all_table(table):
    def _accepted_text(p):
        text = ""
        for run in p.xpath("w:r[not(w:pPr/w:rPr/w:moveFrom)] | w:ins/w:r"):
            for child in run:
                for child in run:
                    if child.tag == qn("w:t"):
                        text += child.text or ""
                    elif child.tag == qn("w:tab"):
                        text += "\t"
                    elif child.tag in (qn("w:br"), qn("w:cr")):
                        text += "\n"
                    elif child.tag == qn("mc:AlternateContent"):
                        for nested_p in child.iter(qn("mc:Choice")):
                            text += _accepted_text(nested_p)
                            text += "\n"
        return text

    table_text = ""
    for row in table._element.iter(qn("w:tr")):
        for cell in row.iter(qn("w:tc")):
            for p in cell.iter(qn("w:p")):
                table_text += _accepted_text(p)
                table_text += "\t"
        table_text = table_text.rstrip("\t")
        table_text += "\n"
    return table_text.rstrip("\n")


def display_row(row):
    for idx, cell in enumerate(row.cells):
        print(f"cell {idx}: {cell.text}")
