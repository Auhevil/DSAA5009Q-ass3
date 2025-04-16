from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "/data/zli/workspace/zli_work/DSAA6000Q_ass3/model/Llama-2-7b-hf"
lora_model_path = "/data/zli/workspace/LLaMA-Factory/saves/llama2-7b/lora/sft-assistant-3200/checkpoint-300"
output_dir = "/data/zli/workspace/zli_work/DSAA6000Q_ass3/model/sft-assistant-3200"

# Load base model
print(f"Loading base model from {base_model_path}...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)

# Load LoRA weights
print(f"Loading LoRA weights from {lora_model_path}...")
model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge LoRA weights into the base model
print(f"Merging LoRA weights into the base model...")
model = model.merge_and_unload()  # This merges LoRA weights into the base model

# Save the merged model
print(f"Saving merged model to {output_dir}...")
model.save_pretrained(output_dir)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.save_pretrained(output_dir)

print("Model merging complete. Merged model saved!")