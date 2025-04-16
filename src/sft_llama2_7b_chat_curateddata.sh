export CUDA_VISIBLE_DEVICES=0,1
FORCE_TORCHRUN=1 llamafactory-cli train /data/zli/workspace/LLaMA-Factory/examples/train_full/llama2_full_sft_curated_data.yaml \
    > /data/zli/workspace/zli_work/DSAA6000Q_ass3/train_logs/llama2-7b/log_curateddata_$(date +"%Y%m%d_%H%M%S").log 2>&1