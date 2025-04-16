import argparse
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def format_prompt_with_chat_template(text, tokenizer):
    """使用模型的chat_template格式化输入文本"""
    # 创建消息列表，包含系统消息和用户消息
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": text}
    ]
    # 应用chat_template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True  
    )
    return formatted_prompt

def main(model_path: str, data_dir: str, batch_size: int, output_path: str):
    # 1. 加载模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 加载本地 JSONL 数据集，并抽取前 150 条指令
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.jsonl",
            "test":  f"{data_dir}/test.jsonl"
        }
    )
    sampled = dataset["train"].shuffle(seed=42).select(range(150))
    prompts = sampled["conversations"]
    inputs = []
    for conv in prompts:
        # 使用chat_template格式化输入
        formatted_prompt = format_prompt_with_chat_template(conv[1], tokenizer)
        inputs.append(formatted_prompt)

    # 3. 切分成若干批次
    prompt_batches = batch_data(inputs, batch_size)

    # 4. 准备 SamplingParams
    # 根据模型的chat_template调整stop_tokens
    stop_tokens = ["</s>"]
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=1024,
        stop=stop_tokens
    )

    # 5. 初始化 vLLM 引擎
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        max_num_seqs=batch_size,
        trust_remote_code=True  # 可能需要加载自定义代码
    )

    # 6. 分批生成并收集结果
    results = []
    for batch_idx, batch_prompts in enumerate(prompt_batches):
        outputs = llm.generate(
            batch_prompts,
            sampling_params=sampling_params
        )
        for original_prompt, output in zip(batch_prompts, outputs):
            generated = output.outputs[0].text
            # 保存原始格式化后的提示和生成的回复
            results.append({
                "formatted_prompt": original_prompt,
                "response(from lima)": sampled["conversations"][len(results)][1],  # 原始响应
                "instruction(generated)": generated
            })
        print(f"[Batch {batch_idx + 1}/{len(prompt_batches)}] 完成，当前累计结果：{len(results)} 条")

    # 7. 保存到 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"全部推理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 对对话数据并行推理")
    parser.add_argument(
        "--model_path", type=str,
        default="/data/zli/workspace/zli_work/DSAA6000Q_ass3/model/sft-assistant-all",
        help="model path"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="../data/lima",
        help="本地数据文件夹，包含 train.jsonl 和 test.jsonl"
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=10,
        help="并行推理时每批次的大小"
    )
    parser.add_argument(
        "--output", type=str,
        default="../results/selfaugmentation_result.json",
        help="推理结果保存路径"
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        output_path=args.output
    )