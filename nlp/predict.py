import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    logits = outputs.logits
    probabilities = softmax(logits, dim=-1)
    pest_probability = probabilities[0][1].item()
    return pest_probability

# 示例用法
if __name__ == "__main__":
    model_path = "trained_model"  # 确保这是你的模型存储路径
    model, tokenizer = load_model(model_path)

    text_to_classify = "The image features a group of ants walking on a wooden surface..."
    probabilities = predict(text_to_classify, model, tokenizer)
    print("Probabilities:", probabilities)
