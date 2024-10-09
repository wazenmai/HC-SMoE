# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

# modified
# @Author: wazenmai
# @Time: 2024/8/13

import os
import sys
import json
import time
import torch
import pickle
import logging
import itertools
from fire import Fire
from tqdm import tqdm
from typing import Optional
from transformers import MixtralForCausalLM, AutoTokenizer

from hcsmoe.evaluation import evaluate_fewshot, get_calib_dataloder
from hcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral, FineGrainedExpertsGrouperForMixtral
from hcsmoe.merging.grouping_mixtral import merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models, merge_by_feature_selection
from hcsmoe.models.model_wrapper import PrunableMixtralSparseMoeBlockWrapper
from hcsmoe.models.mixtral.modeling_mixtral import MyMixtralForCausalLM
from hcsmoe.utils.helper import save_json

logger = logging.getLogger(__name__)

from typing import Optional


###############################
#  Helper class and functions #
###############################
class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "zipit",
        mode: Optional[str] = "normal",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        try_oracle: Optional[bool] = False,
        random_start_center: Optional[bool] = False,
        weight= None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        overlap_metric: Optional[str] = "cosine",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.try_oracle = try_oracle
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=4,
    )

def get_grouper(args, config):
    return ExpertsGrouperForMixtral(
        config=config,
        similarity_base=args.similarity_base,
        start_layer=args.start_layer,
        group_limit=args.group_limit,
        data_limit=args.data_limit,
        random_start_center=args.random_start_center,
        cluster=args.cluster,
        linkage=args.linkage,
        hierarchical_stopping_metric=args.hierarchical_stopping_metric,
        overlap_metric=args.overlap_metric,
        dynamic_group=args.dynamic_group,
    )

def evaluation(args, model, tokenizer):
    result_dir = args.result_path.split("/")[:-1]
    result_dir = "/".join(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(args.task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=args.task, num_fewshot=args.num_fewshot, output_path=args.result_path, log=True
        )
    else:
        for i, t in enumerate(args.tasks):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=args.num_fewshot, eval_batch_size=args.eval_batch_size, output_path=args.result_path, log=True
            )

def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()


###############################
###      Main function      ###
###############################
def run_hcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge", # random, frequency, knowledge, no
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, kl-weight, fix-dom-same
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        random_start_center: Optional[bool] = False,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        ingredient: Optional[str] = "act", # act, weight, act+weight
        overlap_metric: Optional[str] = "cosine", # kl-divergence, wasserstein, cosine,
        dynamic_group: Optional[bool] = False,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} {mode} merge with ingredient {ingredient}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}, hierarchical_stopping_metric: {hierarchical_stopping_metric}, overlap_metric: {overlap_metric}, group: {group}, dynamic_group: {dynamic_group}")


    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        hierarchical_stopping_metric=hierarchical_stopping_metric,
        overlap_metric=overlap_metric,
        dynamic_group=dynamic_group,
    )
    
    torch.manual_seed(0)
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader_for_merging = get_dataloader(args, tokenizer)

    grouper = get_grouper(args, model.config)

    
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters())
    print(f"[HC-SMoE] Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge == "freq" or dominant == "frequency" or mode == "freq":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    if dynamic_group:
        grouper.compute_all_usages(model, dataloader_for_merging, mode=hierarchical_stopping_metric)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    

    ### 2. Get dominant experts
    dom_experts = None
    if merge == "fsm" or merge == "no":
        pass
    elif dominant == "random":
        grouper.group_experts_randomly(num_groups=args.num_average_groups)
        dom_experts = None
    elif dominant == "frequency":
        grouper.compute_all_similarities(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "routing-score":
        grouper.compute_all_usages(model, dataloader_for_merging, mode="routing-score")
        print_usage_frequency(grouper._usage_frequency_state_dict)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "knowledge":
        grouper.compute_all_similarities(model, dataloader_for_merging)
        model = grouper.all_in_one_knowledge_dominant(
            model=model, 
            dataloader=dataloader_for_merging,
            merge=merge,
            mode=mode,
            num_groups=num_average_groups,
            dominant_alone=False,
            usage_weighted=False,
        )
        dom_experts = grouper.core_experts
    elif dominant == "no":
        dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
    else:
        raise ValueError(
            f"Accepted dominant are `random`, `frequency`, `knowledge`, `no`, but the input is `{dominant}`")

    
    ### 3. Merge experts
    if merge == "no":
        pass
    elif group == "module":
        model = grouper.merge_weighted(model, merge, core_experts=dom_experts)
    elif dominant != "knowledge":
        if merge == "freq":
            model = merge_by_groups_with_usage_weighted(
                model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
            )
        elif merge == "fsm":
            model, dom_experts = merge_by_feature_selection(
                model, grouper=grouper, dataloader=dataloader_for_merging, num_groups=num_average_groups, mode=mode
            )
        else:
            model = merge_by_groups_within_and_across_models(
                mixtral_model=model,
                grouper=grouper,
                dataloader=dataloader_for_merging,
                merge=merge,
                mode=mode,
                partition=partition,
                dominant_alone=False,
                core_experts=dom_experts,
                ingredient=ingredient,
            )
    
    print(f"[HC-SMoE] Merging time: {time.time() - group_st:2f} seconds")
    
    
    if merge != "no":
        ### 4. Print grouping results
        print(f"[HC-SMoE] ========= Grouping results ========= ")
        if group == "module":
            for name, state in grouper.group_state_dict().items():
                print(f"Group {name} - ", end='')
                for module_name, label in state.items():
                    print(f"{module_name}: {label.tolist()}, ", end='')
                print()
        else:
            for name, state in grouper.group_state_dict().items():
                if dom_experts is None:
                    print(f"Group {name}: {state.tolist()}")
                else:
                    print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")
        del grouper

    if merge == "unmerge":
        print(f"[HC-SMoE] ======= Grouping of unmerge ======= ")
        for layer_idx in range(start_layer, model.config.num_hidden_layers):
            print(f"--- Layer {layer_idx} ---")
            print(f"expert_to_group: {model.model.layers[layer_idx].block_sparse_moe.expert_to_group}")
            print(f"group_to_expert: {model.model.layers[layer_idx].block_sparse_moe.group_to_expert}")
            print(f"unmerge_matrix: {model.model.layers[layer_idx].block_sparse_moe.unmerge_matrix}")
    
    ### 5. Save the model
    print("[HC-SMoE] Number of parameters after merging:", model.num_parameters())
    if num_average_groups < model.config.num_experts_per_tok:
        model.config.num_experts_per_tok = num_average_groups
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(model.state_dict(), output_path+"/model.pth")
    torch.cuda.empty_cache()

    ### 6. Evaluation
    evaluation(args, model, tokenizer)

if __name__ == "__main__":
    Fire(run_hcsmoe)
