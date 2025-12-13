from numpy import exp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from typing import Any
from datasets import load_from_disk


def get_dataset(least_length=10):

    test_dataset = load_from_disk("./dataset/wikitext/test")
    texts: list[str] = test_dataset["text"]
    filtered_text = []
    for text in texts:
        text = text.strip()
        if text == '':
            continue
        if text[0] == '=' and text[-1] == '=':
            continue
        if len(text) < least_length:
            continue
        filtered_text.append(text)

    return filtered_text

def get_loss_and_length(input_text: str, model: Any, tokenizer: Any) -> tuple[float, int]:

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs_with_labels = model(**inputs, labels=inputs["input_ids"])
        loss = outputs_with_labels.loss.item()

    # multiplied by length - 1
    length = len(inputs["input_ids"][0]) - 1
    return loss * length, length

def get_perplexity(loss: list[float], length: list[int]):
    average_loss = sum(loss) / sum(length)
    return exp(average_loss)

def main():
    model_path = "/path/to/your/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to("cuda")

    test_dataset = get_dataset()
    loss_list, length_list = [], []
    for data in tqdm(test_dataset):
        loss, length = get_loss_and_length(data, model, tokenizer)        
        loss_list.append(loss)
        length_list.append(length)

    perplexity = get_perplexity(loss_list, length_list)
    print(f"Average Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
