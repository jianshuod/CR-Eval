import torch
from typing import List, Iterator
from torch.utils.data import Sampler


class StageSampler(Sampler[List[int]]):
    r"""Samples elements from a list of lists of indices.

    Args:
        stage_indices (List[List[int]]): A list of lists where each inner list contains indices to sample from.
        generator (torch.Generator, optional): Generator used in sampling. If None, Python's random module will be used.
    """

    def __init__(self, stage_indices: List[List[int]], generator=None) -> None:

        self.stage_indices = stage_indices
        self.generator = generator

        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError(
                "generator should be an instance of torch.Generator or None"
            )

    def __iter__(self) -> Iterator[List[int]]:
        for indices in self.stage_indices:
            for i in torch.randperm(len(indices), generator=self.generator):
                yield indices[i]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.stage_indices)
