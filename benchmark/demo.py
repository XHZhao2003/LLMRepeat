import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/home/zxh/zxh/LLM_models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to("cuda")

text = ["The quick brown fox jumps over the lazy dog."]
inputs = tokenizer(text, return_tensors="pt").to("cuda")


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, T, vocab_size)``

input_ids = inputs["input_ids"][0] # shape: (T,)
T = input_ids.size(0)

# 计算 softmax 概率（仅对前 T-1 个位置，因为最后一个位置无预测目标）
probs = torch.softmax(logits[0, :-1, :], dim=-1)  # shape: (T-1, vocab_size)

print(f"Tokenized input: {tokenizer.convert_ids_to_tokens(input_ids)}\n")
print(f"{'Position':<8} {'Predicted -> Target':<30} {'Prob':<12} {'-log(p)':<10}")
print("-" * 65)

total_nll = 0.0
for i in range(T - 1):
    target_token_id = input_ids[i + 1].item()
    predicted_prob = probs[i, target_token_id].item()
    nll = -torch.log(torch.tensor(predicted_prob)).item()  # negative log-likelihood
    total_nll += nll

    target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]
    # 当前位置的 logits 是基于 input_ids[0..i] 预测 target_token (即 input_ids[i+1])
    print(f"{i:<8} {'-> ' + repr(target_token):<30} {predicted_prob:<12.6f} {nll:<10.4f}")

# 验证：平均 NLL 应等于 model(..., labels=...) 的 loss
avg_nll = total_nll / (T - 1)
perplexity_manual = torch.exp(torch.tensor(avg_nll)).item()

# 对比官方 loss
with torch.no_grad():
    outputs_with_labels = model(**inputs, labels=inputs["input_ids"])
    loss_official = outputs_with_labels.loss.item()
    perplexity_official = torch.exp(torch.tensor(loss_official)).item()

print("\n" + "="*65)
print(f"Manual average NLL:     {avg_nll:.6f}")
print(f"Official loss:          {loss_official:.6f}")
print(f"Manual perplexity:      {perplexity_manual:.2f}")
print(f"Official perplexity:    {perplexity_official:.2f}")