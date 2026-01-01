### Perplexity

On wikitest-103-v1 dataset

|Llama-3.2-3B|Qwen-3-0.6B|Qwen-3-1.7B|Qwen-3-4B|DeepSeek-R1-Distill-Qwen-1.5B|
|--|--|--|--|--|
|13.682|37.932|28.705|26.094|92.557|


### Rep-3

On MMLU dataset

```python
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=args.max_new_tokens,
    ignore_eos=True
)
```
```shell
python evaluate_mmlu.py \
    --model_path ../LLM_models/DeepSeek-R1-Distill-Qwen-1.5B \
    --parquet_file dataset/mmlu/test.parquet \
    --output_file output/DeepSeek-R1-Distill-Qwen-1.5B-mmlu.jsonl \
    --max_new_tokens 256
```

|Llama-3.2-3B|Qwen-3-0.6B|Qwen-3-1.7B|Qwen-3-4B|DeepSeek-R1-Distill-Qwen-1.5B|
|--|--|--|--|--|
|0.6808|0.7231|0.6559|0.3277|0.5287|


-----------------------------------------------------------------

```python
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=args.max_new_tokens,
    ignore_eos=True
)
```
```shell
python evaluate_mmlu.py \
    --model_path ../LLM_models/Qwen3-4B \
    --parquet_file dataset/mmlu/test.parquet \
    --output_file output/Qwen3-4B-mmlu.jsonl \
    --add_prompt \
    --max_new_tokens 256
```


|Llama-3.2-3B|Qwen-3-0.6B|Qwen-3-1.7B|Qwen-3-4B|DeepSeek-R1-Distill-Qwen-1.5B|
|--|--|--|--|--|
|0.6380|0.7124|0.4826|0.4524|0.1910|