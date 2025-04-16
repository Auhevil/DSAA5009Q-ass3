import argparse
from transformers import AutoTokenizer
import json
from datasets import load_dataset
import re
from vllm import LLM, SamplingParams
import sys
import os
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

prompt_for_curated = '''
Evaluate the quality of the AI Assistant's response to the user's instruction using this 5-point scale:

Score 1: The response is inadequate— incomplete, vague, off-topic, or failing to address the user's request. It may include irrelevant content, personal experiences, promotional material, or resemble a blog or forum post rather than an AI Assistant's answer.

Score 2: The response is partially helpful but does not fully meet the user's needs. It may address some aspects of the instruction while missing others or offer a general approach without specific details.

Score 3: The response is helpful and complete but lacks an AI Assistant's tone. It may seem like an excerpt from a blog, web page, or search results, with personal opinions or references to external content.

Score 4: The response is well-crafted from an AI Assistant's perspective, providing a complete, clear, and comprehensive answer. It is organized and helpful, with only minor issues like slight verbosity or lack of focus.

Score 5: The response is outstanding, perfectly addressing the instruction with expert knowledge. It is logical, engaging, easy to follow, and free of irrelevant content or errors.

First, briefly explain your reasoning for the score, then write 'Score: [number]' on the last line.

Instruction: {generated_instruction}
Answer: {ground_truth_output}

Assistant: 
'''


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
    # load data from self-augmented dataset
    input_file = "../data/selfaugmentation_result.json"
    output_file = "../data/score_for_curated.json"
    with open(input_file, "r") as f:
        augmented_data = json.load(f)
    inputs = []
    # add chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for data in augmented_data:
        inputs.append({
            "prompt": format_prompt_with_chat_template(prompt_for_curated.format(
                generated_instruction=data["instruction(generated)"],
                ground_truth_output=data["response(from lima)"]
            ), tokenizer),
            "instruction(generated)": data["instruction(generated)"],
            "response(from lima)": data["response(from lima)"]
        })
    print(f"len(inputs): {len(inputs)}")
    print(f"example): {inputs[0]}")

    # Check if output file exists and load existing results
    results = []
    processed_prompts = set()
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_prompts = {item["formatted_prompt"] for item in results}
                print(f"Loaded {len(results)} existing results from {output_file}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing output file. Starting from scratch.")
            results = []
    
    # Filter out already processed prompts
    remaining_inputs = [prompt for prompt in inputs if prompt["prompt"] not in processed_prompts]
    print(f"Remaining prompts to process: {len(remaining_inputs)}/{len(inputs)}")
    
    if not remaining_inputs:
        print("All prompts have been processed. Nothing to do.")
    else:
        # form batch data
        prompt_batches = batch_data(remaining_inputs, batch_size)  # Use remaining_inputs instead of inputs
        # load model
        stop_tokens = ["</s>"]
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=1024,
            stop=stop_tokens
        )
        llm = LLM(
            model=model_path,
            tensor_parallel_size=2,
            max_num_seqs=batch_size,
            trust_remote_code=True  # 可能需要加载自定义代码
        )

        # generate
        results = []
        for batch_idx, batch_items in enumerate(prompt_batches):
            # Extract only the prompts for generation
            batch_prompts = [item["prompt"] for item in batch_items]
            
            outputs = llm.generate(
                batch_prompts,
                sampling_params=sampling_params
            )
            
            for batch_item, output in zip(batch_items, outputs):
                generated = output.outputs[0].text
                # Save with additional instruction and response fields
                results.append({
                    "formatted_prompt": batch_item["prompt"],
                    "answer": generated,
                    "instruction(generated)": batch_item["instruction(generated)"],
                    "response(from lima)": batch_item["response(from lima)"]
                })
            print(f"[Batch {batch_idx + 1}/{len(prompt_batches)}] finished, total: {len(results)}")
        # store the output
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"result of score are stored at {output_path}")
    # analyse the answer file and select curated data file
    with open(output_file, "r") as f:
        results_from_file = json.load(f)
    curated_list = []
    for item in results_from_file:
        text = item["answer"]
        match = re.search(r"Score:\s*(\d+)", text)
        score = int(match.group(1)) if match else 0
        if score > 3:
            curated_list.append(item)
    curated_file = "../data/curated_file.json"
    curated_data = []
    for item in curated_list:
        # Remove the score from the answer
        input = item["instruction(generated)"]
        output = item["response(from lima)"]
        curated_data.append(
            {
                "input": input,
                "output": output
            }
        )
    # Write the curated list to file
    with open(curated_file, "w", encoding="utf-8") as f:
        json.dump(curated_data, f, ensure_ascii=False, indent=2)
    print(f"Curated data saved: {len(curated_list)} items with positive scores")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 对对话数据并行推理")
    parser.add_argument(
        "--model_path", type=str,
        default="/data/zli/workspace/zli_work/DSAA6000Q_ass3/model/Llama-2-7b-chat-hf",
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
