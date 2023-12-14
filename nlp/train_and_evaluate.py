import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_preparation import load_and_preprocess_data

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
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

    # 创建损失函数实例
    loss_function = CrossEntropyLoss()

    # 训练模型
    model.train()
    for epoch in range(3):  # 迭代次数可以调整
        for batch in tqdm(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            # 清除之前的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(**inputs)

            # 计算损失
            loss = loss_function(outputs.logits, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

    # 评估模型
    model.eval()
    predictions, true_labels = [], []
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")

    # 保存模型
    model.save_pretrained(model_path)

# 确保其他文件（data_preparation.py）也正确配置
