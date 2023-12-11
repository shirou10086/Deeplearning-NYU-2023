# train_and_evaluate.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_preparation import load_and_preprocess_data

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate(dataset_path, model_path, num_labels):
    # 加载和预处理数据
    train_encodings, train_labels, test_encodings, test_labels = load_and_preprocess_data(dataset_path)

    # 创建数据集
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 训练模型
    model.train()
    for epoch in range(3):  # 可以调整迭代次数
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 评估模型
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)

    # 保存模型
    model.save_pretrained(model_path)
