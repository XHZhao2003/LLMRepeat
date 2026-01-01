python evaluate_mmlu.py \
    --model_path ../LLM_models/Qwen3-4B \
    --parquet_file dataset/mmlu/test.parquet \
    --output_file output/Qwen3-4B-mmlu.jsonl \
    --add_prompt \
    --max_new_tokens 256
