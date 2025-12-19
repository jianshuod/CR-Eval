from typing import List
from pathlib import Path
import re, os, json, subprocess
from src.utils.logging import logging
from src.make_data.instance import RetrievalHit
# from pyserini.search.lucene import LuceneSearcher
from src.make_data.play_with_spec import SpecContextManager
from src.make_data.spec_processing.chunking import build_documents
from src.configs import CACHE_INDEX_DIR, CACHE_INSTANCE_DIR, CACHE_CHUNK_DIR

logger = logging.getLogger(__name__)


def getReference(text) -> str:
    self_ref_re_exp = r"(?<=clauses\s)[\d\.]+"
    other_ref_re_exp = r"(?<=3GPP\sTS\s)\d*?\.\d*?(?=\s\[\d*?\])"
    self_match_iter = re.finditer(self_ref_re_exp, text)
    other_match_iter = re.finditer(other_ref_re_exp, text)
    return {
        "self": [match.group().rstrip(".") for match in self_match_iter],
        "other": [match.group().rstrip(".") for match in other_match_iter],
    }


available_retrieval_modes = [
    "bm25",
    "tf-idf",  # TODO: To be add
    "dense",  # TODO: To be add
]


class RetrievalManager:

    def __init__(
        self,
        spec_manager: SpecContextManager,
        chunking_mode: str,
        retrieval_mode: str = "bm25",
    ) -> None:
        self.spec_manager = spec_manager
        self.chunking_mode = chunking_mode
        self.retrieval_mode = retrieval_mode

        # urge the spec_manager to apply the chunks
        self.spec_manager.clear_chunks()
        self.spec_manager.apply_chunks(self.chunking_mode)

    def make_index(
        self,
        document_encoding_func,
        instance_id,
    ):
        """
        Builds an index for a given set of documents using Pyserini.

        Args:
            document_encoding_func (function): The function to use for encoding documents.
            instance_id (int): The ID of the current instance.

        Returns:
            index_path (Path): The path to the built index.
        """
        index_path = Path(CACHE_INDEX_DIR, f"index__{str(instance_id)}", "index")
        if index_path.exists():
            self.index_path = index_path
            return index_path
        thread_prefix = f"(pid {os.getpid()}) "

        documents_path = Path(CACHE_INSTANCE_DIR, instance_id, "documents.jsonl")
        if not documents_path.parent.exists():
            documents_path.parent.mkdir(parents=True)
        documents = build_documents(
            self.spec_manager.list_chunks(), document_encoding_func
        )
        with open(documents_path, "w") as docfile:
            for relative_path, contents in documents.items():
                print(
                    json.dumps({"id": relative_path, "contents": contents}),
                    file=docfile,
                    flush=True,
                )
        cmd = [
            subprocess.check_output(["which", "python"])
            .decode("utf-8")
            .strip(),  # path to `python` bin
            "-m",
            "pyserini.index",
            "--collection",
            "JsonCollection",
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "2",
            "--input",
            documents_path.parent.as_posix(),
            "--index",
            index_path.as_posix(),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            output, error = proc.communicate()
        except KeyboardInterrupt:
            proc.kill()
            raise KeyboardInterrupt
        if proc.returncode == 130:
            logger.warning(thread_prefix + f"Process killed by user")
            raise KeyboardInterrupt
        if proc.returncode != 0:
            logger.error(f"return code: {proc.returncode}")
            raise Exception(
                thread_prefix
                + f"Failed to build index for {instance_id} with error {error}"
            )

        self.index_path = index_path
        return index_path

    def search(self, instance: dict, field) -> List[RetrievalHit]:
        """
        Use different fields from the instance to query different contexts,
        and then return Union(results)

        Return `hits`

        """
        instance_id = instance["instance_id"]
        if not self.index_path:
            logger.error(f"Index of {instance_id} not found")
            return

        try:
            searcher = LuceneSearcher(self.index_path.as_posix())
            cut_off = len(instance[field])
            while True:
                try:
                    hits = searcher.search(
                        instance[field][:cut_off],
                        k=20,
                        remove_dups=True,
                    )
                except Exception as e:
                    if "maxClauseCount" in str(e):
                        cut_off = int(round(cut_off * 0.8))
                        continue
                    else:
                        raise e
                break
            results = {"instance_id": instance_id, "hits": []}
            for hit in hits:
                results["hits"].append(
                    {
                        "doc_id": hit.docid,
                        "doc_content": self.load_spec_content(hit.docid),
                        "score": hit.score,
                    }
                )
            return results["hits"]
        except Exception as e:
            logger.error(f"Failed to process {instance_id}")

    def load_spec_content(self, chunk_id):
        """
        Load the content of the spec with the `chunk_id` from the local storage
        `chunk_id`is the position_anchor used in the `build_documents` function
        """

        with open(Path(CACHE_CHUNK_DIR, chunk_id), "r", encoding="utf-8") as doc:
            content = doc.read()
        return content

    def get_referenced_section(self, content):
        # Extract referenced content
        ref_section_list = getReference(content)["self"]
        ref_rgexp = "|".join(
            [i.replace(".", r"\.") + r"\s.*" for i in ref_section_list]
        )

        references = {}
        for abs_path in self.spec_manager.list_chunks():
            if re.search(ref_rgexp, abs_path):
                references[abs_path] = self.load_spec_content(abs_path)

        return references
