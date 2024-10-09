export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="your-huggingface-home-path"

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="frequency" \
  --similarity_base="expert-output" \
  --merge="freq" \
  --mode="normal" \
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/merge-45e/test" |& tee results/log_45e_test