import logging
import numpy as np
import collections
from tqdm.auto import tqdm
from itertools import chain
from typing import Iterable
from collections import Counter
from stringology.ac import AhoCorasick
from stringology.ngrams import all_ngrams
from multiprocessing import cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console handler
    ],
)
logger = logging.getLogger(__name__)

from tqdm.auto import tqdm

def list_split(lst, sections):
    """
    Splits a list into N sections efficiently using numpy array operations.
    """
    arr = np.array(lst)
    splits = np.array_split(arr, sections)
    return [split.tolist() for split in splits]


class OverlapyTestSet:
    def __init__(self, name, min_n=8, max_n=13, percentile=5, examples=None):
        assert isinstance(min_n, int) and isinstance(max_n, int)
        assert 1 <= min_n <= max_n
        assert 0 <= percentile <= 100
        self.name = name
        self.min_n = min_n
        self.max_n = max_n
        self.percentile = percentile
        self.examples = examples or []

    def add_example(self, example):
        self.examples.append(example)

    @staticmethod
    def get_percentile(values, percentile):
        return np.percentile(values, percentile)

    def compute_n(self):
        """
        Compute the optimal size of N-Grams for data contamination studies, for this testset.
        """
        hist = [len(ex) for ex in self.examples]
        n = int(self.get_percentile(hist, self.percentile))
        return min(max(self.min_n, n), self.max_n)

    def ngrams(self):
        """
        Compute ngrams of size N (see compute_n()) for each example.
        """
        n = self.compute_n()
        for example in self.examples:
            yield from all_ngrams(example, minn=n, maxn=n)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    def get_matches(self, matches):
        """
        Given a dictionary of matches, retrieve the matched examples efficiently.
        """
        ac = AhoCorasick(matches.keys())
        for i, example in enumerate(self.examples):
            yield from ((i, ngram, pos) for ngram, pos in ac(example))


class OverlapyNgramMatcher:
    def __init__(self, ngrams: set):
        self.ac = AhoCorasick(ngrams)

    def __call__(self, examples):
        matches = collections.defaultdict(list)
        for i, example in enumerate(examples):
            for ngram, _ in self.ac(example):
                matches[ngram].append(i)
        return matches


class Overlapy:
    def __init__(self, testsets: list, dataset: Iterable, n_workers=cpu_count()):
        """
        Initialize Overlapy with testsets, a dataset, and number of workers.
        """
        assert 1 <= n_workers <= cpu_count(), "Invalid number of workers"
        self.dataset = list(dataset)
        self.testsets = testsets
        self.n_workers = n_workers
        self.testset_ngrams = set(
            map(tuple, chain(*[list(testset.ngrams()) for testset in testsets]))
        )

    def _worker_function(self, args):
        chunk_idxs, testset_ngrams, worker_id = args
        matcher = OverlapyNgramMatcher(testset_ngrams)
        matches = collections.defaultdict(list)

        for idx in chunk_idxs:
            matched = matcher([self.dataset[idx]])
            for ngram, _ in matched.items():
                matches[ngram].append(idx)

        return matches

    def run(self):
        """
        Perform parallel matching of ngrams between dataset and testsets.
        """
        dataset_indices = list(range(len(self.dataset)))
        chunks = list_split(dataset_indices, self.n_workers)

        manager = Manager()
        shared_ngrams = manager.list(self.testset_ngrams)
        worker_args = [(chunk, shared_ngrams, i) for i, chunk in enumerate(chunks)]

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._worker_function, args) for args in worker_args
            ]
            results = []
            for f in tqdm(futures, desc="Processing chunks"):
                results.append(f.result())

        combined_matches = collections.defaultdict(list)
        for result in results:
            for ngram, positions in result.items():
                combined_matches[ngram].extend(positions)

        return combined_matches


class OverlapFinder:
    def __init__(self, testsets: list, trainingset: Iterable, n_workers=cpu_count()):
        """
        Initializes the OverlapFinder with optimized data structures.
        """
        self.trainingset = list(trainingset)
        self.testsets = testsets
        self.n_workers = n_workers
        self.testset_ngrams = set(
            map(tuple, chain(*[list(testset.ngrams()) for testset in testsets]))
        )
        self.matcher = OverlapyNgramMatcher(self.testset_ngrams)

    def _process_chunk(self, chunk_idxs):
        """
        Process a chunk of training data efficiently.
        """
        matches = collections.defaultdict(list)
        for idx in chunk_idxs:
            matched = self.matcher([self.trainingset[idx]])
            for ngram, _ in matched.items():
                matches[ngram].append(idx)
        return matches

    def find_overlaps(self):
        """
        Finds overlaps between testsets and trainingset using parallel processing.
        """
        # Split work into chunks
        chunks = list_split(range(len(self.trainingset)), self.n_workers)

        # Process chunks in parallel using ThreadPoolExecutor
        combined_matches = collections.defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]

            # Collect results with progress tracking
            for future in tqdm(futures, desc="Processing chunks"):
                result = future.result()
                for ngram, indices in result.items():
                    combined_matches[ngram].extend(indices)

        # Find overlaps for each testset example
        overlaps = collections.defaultdict(set)  # Using set for faster deduplication
        ac = AhoCorasick(combined_matches.keys())

        for testset_idx, testset in enumerate(self.testsets):
            for i, example in enumerate(testset.examples):
                for ngram, _ in ac(example):
                    overlaps[i].update(combined_matches[ngram])

        # Convert sets to sorted lists for consistent output
        return {k: sorted(list(v)) for k, v in overlaps.items()}

    def print_overlaps(self):
        """
        Print overlaps with improved formatting and error handling.
        """
        try:
            logger.info("Finding overlaps...")
            overlaps = self.find_overlaps()

            logger.info("\n=== Overlap Report ===")
            if not overlaps:
                logger.info("No overlaps found.")
                return

            # Track valid overlaps for accurate counting
            valid_overlap_count = 0
            valid_testset_examples = 0

            counter = Counter()

            for testset_idx, training_indices in tqdm(
                overlaps.items(), desc="Printing overlaps"
            ):
                has_valid_overlap = False
                logger.info(f"\nTestset Example {testset_idx}:")
                logger.info("-" * 50)
                logger.info("Testset content:")
                testset_content = " ".join(self.testsets[0].examples[testset_idx])
                logger.info(testset_content)
                logger.info("\nOverlapping Training Examples:")

                for train_idx in training_indices:
                    try:
                        logger.info(f"\nTraining Example {train_idx}:")
                        training_content = " ".join(self.trainingset[train_idx])
                        logger.info(training_content)

                        # Find and print exact overlapping ngrams
                        n = self.testsets[0].compute_n()
                        # Convert ngrams to tuples to make them hashable
                        testset_ngrams = set(
                            tuple(ngram)
                            for ngram in all_ngrams(
                                self.testsets[0].examples[testset_idx], minn=n, maxn=n
                            )
                        )
                        training_ngrams = set(
                            tuple(ngram)
                            for ngram in all_ngrams(
                                self.trainingset[train_idx], minn=n, maxn=n
                            )
                        )
                        overlapping_ngrams = testset_ngrams.intersection(
                            training_ngrams
                        )

                        if overlapping_ngrams:
                            filtered_ngrams = []
                            for ngram in list(overlapping_ngrams):
                                filtered_ngrams.append(ngram)

                            # Only print and count if there are valid overlaps after filtering
                            if filtered_ngrams:
                                has_valid_overlap = True
                                valid_overlap_count += len(filtered_ngrams)
                                for ngram in filtered_ngrams:
                                    segment = " ".join(ngram)
                                    logger.info(
                                        f"Exact overlapping ngrams {testset_idx}:\n\n{segment}"
                                    )
                                    counter[testset_idx] += 1
                                logger.info("-" * 30)

                    except Exception as e:
                        logger.error(
                            f"Error processing training example {train_idx}: {str(e)}"
                        )
                        logger.error(f"Testset ngrams type: {type(testset_ngrams)}")
                        logger.error(f"Training ngrams type: {type(training_ngrams)}")
                        logger.error(
                            f"Sample ngram type: {type(next(iter(testset_ngrams))) if testset_ngrams else 'empty'}"
                        )

                if has_valid_overlap:
                    valid_testset_examples += 1

            logger.info("\n=== Overlap Report Summary ===")
            logger.info(
                f"Found {valid_overlap_count:,} valid overlaps across {valid_testset_examples:,} testset examples"
            )
            logger.info("\n=== End of Overlap Report ===")
            print(counter)

        except Exception as e:
            logger.error(f"Error in print_overlaps: {str(e)}")
            raise
