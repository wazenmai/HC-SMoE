# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
from itertools import chain
from typing import Optional

import itertools
import logging
import torch
import transformers
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from transformers.testing_utils import CaptureLogger

logger = logging.getLogger(__name__)

DATASETS = {
    'c4': lambda: load_dataset('json', data_files={'train': 'hcsmoe/data/c4-train.00000-of-01024.json'}, trust_remote_code=True),
}

def get_calib_dataloder(
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    max_block_size: int,
    n_blocks_for_stat: int,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
):
    all_set = DATASETS[dataset]()
    block_size = tokenizer.model_max_length

    if block_size > max_block_size:
        logger.info(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `max_block_size` value"
            f" of {max_block_size}. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--max_block_size xxx`."
        )
        block_size = max_block_size

    if n_blocks_for_stat > 0:  # Random choose `n_blocks_for_stat` blocks
        calib_set = all_set['train'].shuffle(seed=seed).select(
            range(min(n_blocks_for_stat * 16, len(all_set['train']))))
    else:   # Use the whole set
        logger.warning('n_blocks_for_stat <= 0, using the whole dataset.')
        calib_set = all_set['train'].shuffle(seed=seed)
    
    logger.info(f'Calibration dataset: {calib_set}')
    text_column_name = "text" if "text" in calib_set.features else list(
        calib_set.features)[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    tokenized_calib_set = calib_set.map(
        tokenize_function,
        batched=True,
        remove_columns=list(calib_set.features),
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_calib_set = tokenized_calib_set.map(
        group_texts,
        batched=True,
    )

    if n_blocks_for_stat > 0:
        assert len(lm_calib_set) > n_blocks_for_stat
        lm_calib_set = lm_calib_set.select(range(n_blocks_for_stat))

    calib_loader = torch.utils.data.DataLoader(
        lm_calib_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_data_collator
    )

    return calib_loader

def get_minipile_dataloder(
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        batch_size: int,
        subset_ratio: Optional[float] = 1.0,
) -> DataLoader:
    dataset = load_dataset("JeanKaddour/minipile", split="validation")
    column_names = dataset.column_names

    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=False),
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=8,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    dataset = dataset.shuffle()
    dataset = dataset.select(range(int(len(dataset) * subset_ratio)))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    return data_loader


def evaluate_minipile_perplexity(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        log: Optional[bool] = True,
        block_size: Optional[int] = 512,
        batch_size: Optional[int] = 1,
) -> float:
    data_loader = get_minipile_dataloder(
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
    )

    loss_list = []
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating")):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=False)
        loss_list.append(outputs.loss.item())

    ppl = torch.exp(torch.tensor(loss_list).mean()).item()

    if log:
        print(f"Perplexity of MiniPile: {ppl:.2f}")

    return ppl
