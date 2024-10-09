from typing import Optional

import os
import time
import logging
import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer

from hcsmoe.pruning.kprune import KPruner
from hcsmoe.evaluation import get_calib_dataloder, evaluate_fewshot

logger = logging.getLogger(__name__)

def pruning_mixtral(
    task: str,
    constraint: float,
    model_name: str,
    output_path: str,
    lam_pred: Optional[float] = 1.0,
    lam_rep: Optional[float] = 1e-5,
    T: Optional[float] = 2.0,
    train_batch_size: Optional[int] = 4,
    eval_batch_size: Optional[int] = 32,
    reconstruct_batch_size: Optional[int] = 1024,
    start_layer: Optional[int] = 0,
    n_sentences: Optional[int] = 32,
    result_path: Optional[str] = None,
    reconstruct: Optional[bool] = False,
):
    print(f"Prune model with constraint {constraint} on {task}.\n Model: {model_name}\n train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}, lam_pred={lam_pred}, lam_rep={lam_rep}, T={T}")

    whole_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, device_map="auto"
    )
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    dataloader = get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=n_sentences,
        batch_size=train_batch_size,
        num_workers=4,
    )

    pruner = KPruner(
        config=model.config,
        start_layer=start_layer,
        reconstruct_batch_size=reconstruct_batch_size,
        lam_pred=lam_pred,
        lam_rep=lam_rep,
        mu=64.0,
        T=T,
        constraint=constraint,
        reconstruct=reconstruct,
    )

    print(f"[Pruning] Number of parameters before pruning: {model.num_parameters() / 1000000:2f} M")

    model = pruner.kprune_for_mixtral(
        model=model,
        dataloader=dataloader,
    )

    print(f"[Pruning] Number of parameters after pruning: {model.num_parameters() / 1000000:2f} M")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(model.state_dict(), output_path+"/model.pth")
    if isinstance(task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=0, output_path=output_path, log=True
        )
    else:
        tasks = ["openbookqa", "rte", "winogrande", "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu"]
        eval_batch_sizes = [32, 32, 32, 32, 32, 16, 32, 16]
        for i, t in enumerate(tasks):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=0, eval_batch_size=eval_batch_sizes[i], output_path=result_path, log=True
            )
    
    print(f"THE END: {time.time()-whole_start:.2f} sec.")

if __name__ == "__main__":
    Fire(pruning_mixtral)

