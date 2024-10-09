# Argument Document


- `task`: The tasks to evaluate on.
- `num_average_groups`: Average number of experts we want to leave in each layer, the total number of left experts in the model will be `num_average_group` * number of layers.
- `model_name`: The base model you want to evaluate on. We now support two huggingface classes `MixtralForCausalLM` and `Qwen2MoeForCausalLM`.
- `dominant`: The method of choosing dominant expert in each group.
    - `random`: Randomly group experts
    - `frequency`: The experts with highest frequency globally be the dominant experts. The number of experts distribute non-uniformly through layers.
    - `knowledge`: The experts with highest knowledge (proposed in [here](https://github.com/snudm-starlab/K-prune)) in each layer be the dominant experts. The number of experts distribute uniformly through layers.
    - `no`: No criteria to choose dominant experts first, use clustering intead of single-shot grouping. If the merging method needs dominant expert, we pick the one closest to the center of cluster.
- `similarity_base`: Decide the metric used to group experts. Options: `router-logits`, `weight`, and `expert-output`.
- `merge`: The merging method.
    - `no`: No merge, used to evaluate the original model.
    - `freq`: Use frequency-weighted merging.
    - `zipit`: Use ZipIt adapted from [this repository](https://github.com/UNITES-Lab/MC-SMoE).
    - `kl-weight`: Use knowledge-weighted merging.
    - `fix-dom-same`: Use fixed-dominant merging.
    - `weighted`: Use weighted merging, weight is assigned by `mode`.
- `mode`: The mode for merging method.
    - For merging method `zipit`, `fix-dom-same`. There are four modes.
        - `normal`
        - `activation-with-router-logits`: The feature used to compute pairwise correlation is multiplied with router-logits first.
        - `input-weight`: The merging coefficient is `the number of token routed to the expert / total number of tokens` instead of original average merging.
        - `all`: Use both `activation-with-router-logits` and `input-weight`.
    - For merging method `weighted`, the `mode` is set with `x,y` where `x` is the merging coefficient for dominant experts, and `y` is the merging coefficient for non-dominant experts. When performing average-weighted merging, set `mode=1,1`.
- `ingredient`: The ingredient used in `zipit` and `fix-dom-same` merging.
    - `act`: Use activation feature before feeding in $W_{down}$ matrix in expert, the original ingredient.
    - `weight`: Use concat expert weights instead of activation.
    - `act+weight`: Consider both.
- `n_sentences`: Number of sentences in C4 for inference and collecting necessary statistics for grouping and merging.
- `train_batch_size`: Batch size for collecting necessary statistics for grouping and merging, it does not really "train" since we do not update model parameters based on gradient.
- `eval_batch_size`: Batch size for evaluation.
- `partition`: The partition ratio for grouping and merging, used when the gpu memory is not enough, otherwise `1` means no partition.
- `start_layer`: The layer you want to start merging.
- `output_path`: The path that save final model.
- `result_path`: The path that save the evaluation results for `task`.
- `model_path`: The model path that you saved your own model, no need to set when using base model provided on huggingface.
- `group_limit`: The maximum number of experts in a group.
- `data_limit`: The maximum number of tokens used in `zipit` and `fix-dom-same` merging, used when gpu memory is not enough.
- `num_fewshot`: Number of shots you want to evaluate. In paper we all use zero-evaluation.
- `random_start_center`: Used when performing kmeans, it will randomize the initial center each time.
- `cluster`: The clustering method. It only activate when `dominant=no`.
    - `kmeans`: K-means++ algorithm.
    - `hierarchical`: Hierarchical clustering.
- `linkage`: The linkage method used in hierarchical clustering. Option: `single`, `complete`, `average`, `ward`.
- `hierarchical_stopping_metric`: Used when performing non-uniform hierarchical clustering. Currently unsupported.
- `overlap_metric`: A special mode for single-shot grouping on `expert-output`. It calculate the overlapping rate based on some metrics and group experts.
    - `kl-divergence`
    - `wasserstein`
    - `cosine`: The original single-shot grouping method.
- `dynamic_group`: Whether to allow non-uniform experts distribution for hierarchical clustering. If it is set to `True`, the number of experts in each layer is same as using `dominant=frequency`.


For example, if you want to run HC-SMoE on Mixtral 8x7B to turn it to Mixtral 4x7B.
```
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-mixtral.py \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --model_name="mistralai/Mixtral-8x7B-v0.1" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hirarchical" \
  --linkage="average" \
  --merge="freq" \
  --mode="all" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --start_layer=0 \
  --result_path="results/result_mixtral_test.txt" \
  --output_path="results/" |& tee results/log_mixtral_test
```
