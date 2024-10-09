import os
import random
from typing import Optional

import torch
from tqdm import tqdm


def generate_random_group_labels(
        num_experts: int,
        num_groups: int,
) -> torch.Tensor:
    """
    Assign random group labels to experts, with each group has at least one expert.

    Examples
    --------
    >>> generate_random_group_labels(10, 3).unique().sort()[0]
    tensor([0, 1, 2])
    """
    group_labels = torch.zeros(num_experts, dtype=torch.long)
    for i in range(num_groups):
        group_labels[i] = i
    for i in range(num_groups, num_experts):
        group_labels[i] = random.randint(0, num_groups - 1)
    return group_labels[torch.randperm(num_experts)]
