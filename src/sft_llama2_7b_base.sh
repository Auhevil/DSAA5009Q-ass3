export CUDA_VISIBLE_DEVICES=0
FORCE_TORCHRUN=1 llamafactory-cli train /data/zli/workspace/LLaMA-Factory/examples/train_lora/llama2_lora_sft.yaml \
    > /data/zli/workspace/zli_work/DSAA6000Q_ass3/train_logs/llama2-7b/log_openassistant_$(date +"%Y%m%d_%H%M%S").log 2>&1