import json
import argparse
from numpy import average
from vllm import LLM, SamplingParams

from utils import read_mmlu_parquet


def compute_rep_3(generated_text: str) -> float:
    """
    计算 generated_text 的 rep-3 指标（3-gram 重复率）。
    
    rep-3 = 1 - (unique 3-grams / total 3-grams)
    
    Args:
        generated_text (str): 模型生成的文本。
    
    Returns:
        float: rep-3 值。若文本不足 3 个词，返回 0.0（无重复）。
    """
    if not generated_text:
        return 0.0

    # 按空格分词（适用于英文）
    tokens = generated_text.split()
    
    if len(tokens) < 3:
        return 0.0

    # 生成所有 3-gram
    trigrams = []
    for i in range(len(tokens) - 2):
        trigram = tuple(tokens[i:i+3])
        trigrams.append(trigram)
    
    total_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    
    if total_trigrams == 0:
        return 0.0
    
    rep_3 = 1.0 - (unique_trigrams / total_trigrams)
    return rep_3

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径或 Hugging Face 模型 ID")
    parser.add_argument("--parquet_file", type=str, required=True, help="MMLU parquet 文件路径")
    parser.add_argument("--output_file", type=str, default="mmlu_generations.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--add_prompt", action="store_true", help="启用增强的prompt来减少重复")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（0 表示 greedy）")
    parser.add_threshold_group = parser.add_argument_group("Decoding parameters")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda", help="设备：'cpu', 'cuda', 'auto'")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    args = parser.parse_args()

    # === 读取数据 ===
    records = read_mmlu_parquet(args.parquet_file)
    print(f"Loaded {len(records)} samples from {args.parquet_file}")
    if args.add_prompt:
        add_prompt = "OUTPUT YOUR ANSWER ONLY ONCE AND DO NOT MAKE REPETITION."
        records = [record + add_prompt for record in records]

    llm = LLM(
        model=args.model_path,   # 或 Hugging Face ID
        tokenizer=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=2,
        trust_remote_code=True,
    )

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        ignore_eos=True
    )


    # 批量生成
    outputs = llm.generate(records, sampling_params, use_tqdm=True)

    # 提取结果
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        results.append({
            "index": i,
            "prompt": records[i],
            "generated_text": generated_text,
        })

    # 保存为 JSONL
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    input_jsonl = args.output_file
    all_rep3 = []
    with open(input_jsonl, 'r') as fin:
        for line in fin:
            item = json.loads(line)
            rep3 = compute_rep_3(item["generated_text"])
            all_rep3.append(rep3)

    print(average(all_rep3, axis=0))

if __name__ == "__main__":
    main()