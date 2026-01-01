import pandas as pd

def format_mmlu_prompt(sample):
    """
    将 MMLU 样本转换为适合输入大语言模型的英文用户提示。
    
    Args:
        sample (dict): 包含 'question', 'choices' 的 MMLU 数据项。
                       'choices' 应为长度为 4 的列表或 array，按 [A, B, C, D] 顺序。
    
    Returns:
        str: 格式化的用户提示字符串。
    """
    question = sample['question']
    choices = sample['choices']
    
    # 确保 choices 是一个长度为 4 的序列
    if len(choices) != 4:
        raise ValueError("MMLU choices must have exactly 4 options.")
    
    options_str = "\n".join([f"{label}. {choice}" for label, choice in zip("ABCD", choices)])
    prompt = f"{question}\n{options_str}\nAnswer with the option's letter from the given choices directly."
    
    return prompt

def read_mmlu_parquet(file_path='dataset/mmlu/test.parquet'):
    """
    读取 MMLU 数据集的 Parquet 文件，并将其转换为 prompt 字符串列表。
    
    Args:
        file_path (str): Parquet 文件的路径。
    
    Returns:
        List[str]: 每个元素是一个可直接输入给 LLM 的英文 prompt。
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise RuntimeError(f"无法读取 Parquet 文件 {file_path}: {e}")
    
    records = df.to_dict(orient='records')
    prompts = [format_mmlu_prompt(sample) for sample in records]
    
    return prompts

if __name__ == "__main__":
    dataset = read_mmlu_parquet()
    print(dataset[0])