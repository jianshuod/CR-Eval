import re
import docx
import tiktoken
from tqdm import tqdm
from typing import List
from pathlib import Path
from docx.table import Table
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P


def split_spec_to_chunks(
    spec_path: str, out_dir_path: str, aggregate=False, aggregate_limit=1500
) -> None:
    """
    Split .docx format specification to .txt chunks

    Args:
        spec_path (str): path to the specification file
        out_dir_path (str): path to the output directory
        aggregate (bool, optional): aggregate small-size chunks, defaults to False
        aggregate_limit (int, optional): limit of aggregate size, defaults to 1500

    Returns:
        None
    """

    def num_tokens_from_string(string: str, model_encoding="gpt-4o") -> int:
        """
        Calculate the number of tokens used by a string

        Args:
            string (str): string to calculate the number of tokens for
            model_encoding (str, optional): model encoding, defaults to "gpt-4o"

        Returns:
            int: number of tokens used by the string
        """

        return len(tiktoken.encoding_for_model(model_encoding).encode(string))

    def node_list(parent: docx.document.Document) -> List:
        """
        Get node list of a docx file

        Args:
            parent (docx.document.Document): parent document

        Returns:
            list: node list
        """

        parent_elm = parent.element.body

        nodeList = []
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                nodeList.append(Paragraph(child, parent))
            elif isinstance(child, CT_Tbl):
                nodeList.append(Table(child, parent))
        return nodeList

    def get_table_content(node: Table) -> str:
        """
        Get table content of a table node

        Args:
            node (Table): table node

        Returns:
            str: table content in markdown format
        """

        def check_dup_col(table: List) -> bool:
            """
            Check if the table has duplicate columns

            Args:
                table (list): table to check

            Returns:
                bool: True if the table has duplicate columns, False otherwise
            """

            for row in table:
                if row[::2] != row[1::2]:
                    return False
            return True

        table = [
            [cell.text.replace("\n", "\\n") for cell in row.cells] for row in node.rows
        ]

        if [len(row) for row in table].count(len(table[0])) != len(table):
            temp_table = list()
            for row in table:
                temp_table += row

            new_table = [
                temp_table[i : i + int(len(temp_table) / len(table))]
                for i in range(0, len(temp_table), int(len(temp_table) / len(table)))
            ]

            if [len(row) for row in new_table].count(len(new_table[0])) != len(
                new_table
            ):
                return "| " + " | ".join(temp_table) + " |"

            if check_dup_col(new_table):
                temp_table = [row[::2] for row in new_table]
                new_table = temp_table

            table = new_table

        result = (
            "| "
            + " | ".join(table[0])
            + " |"
            + "\n"
            + "|"
            + ":-:|" * len(table[0])
            + "\n"
            + "\n".join(["| " + " | ".join(row) + " |" for row in table[1:]])
        )

        return result

    def is_image(node) -> bool:
        """
        Check if a node is an image

        Args:
            node: node to check

        Returns:
            bool: True if the node is an image, False otherwise
        """

        results = []
        images = node._element.xpath(".//pic:pic")
        if images and len(images) > 0:
            for image in images:
                results += image.xpath(".//a:blip/@r:embed")

        iter = re.finditer('(?<=(v:imagedata r:id=")).*?(?=")', str(node._element.xml))
        results += [i.group() for i in iter]

        return True if len(results) > 0 else False

    doc = docx.Document(spec_path)
    out_dir_path = Path(out_dir_path)
    out_dir_path.mkdir(exist_ok=True)

    section_list = list()
    start_flag = False
    buffer = list()
    contain_content = False

    for node in tqdm(node_list(doc)):
        try:
            style: str = node.style.name
        except:
            continue

        if (
            (not start_flag)
            and style.startswith("H")
            and style.find("1") != -1
            and (
                "scope" not in node.text.lower()
                and "introduction" not in node.text.lower()
                and "reference" not in node.text.lower()
                and "foreword" not in node.text.lower()
            )
        ):
            start_flag = True
        if not start_flag:
            continue
        if style.startswith("H") and "history" in node.text.lower():
            break

        if style.startswith("H"):
            if len(buffer) > 0 and contain_content:
                section_list.append("\n\n".join(buffer))
                buffer = []
                contain_content = False
            buffer.append(node.text)
        else:
            if isinstance(node, Table):
                content = get_table_content(node)
            else:
                content = node.text
                if style == "TH" and is_image(node):
                    content = ">>> Here is a figure omitted <<<"

            contain_content = True
            buffer.append(content)

    section_list.append("\n\n".join(buffer))

    if aggregate:
        aggregated_sections = [section_list[0]]
        prev = num_tokens_from_string(section_list[0])
        for section in section_list[1:]:
            if prev + num_tokens_from_string(section) > aggregate_limit:
                prev = num_tokens_from_string(section)
                aggregated_sections.append(section)
            else:
                prev += num_tokens_from_string(section)
                aggregated_sections[-1] += "\n\n" + section
        section_list = aggregated_sections

    for id, section in enumerate(section_list):
        out_file_path = out_dir_path / ("%03d.txt" % id)
        with open(out_file_path, "w") as f:
            print(section, file=f)
