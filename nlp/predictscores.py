# predictscores.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取类别1（“有害虫”类别）的概率
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pest_probability = probabilities[:, 1].item()  # 假设类别1是“有害虫”
    # 将概率转换为0到10的分数
    score = np.round(pest_probability * 10, 2)
    return score

# 示例用法
if __name__ == "__main__":
    model_path = "trained_model"
    model, tokenizer = load_model(model_path)

    text_to_classify = "The image features a group of ants walking on a wooden surface..."
    score = predict(text_to_classify, model, tokenizer)
    print("Pest Presence Score:", score)
