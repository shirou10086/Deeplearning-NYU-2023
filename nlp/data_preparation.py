# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_and_preprocess_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2)

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=128)

    # 应用分词
    train_encodings = tokenize_function(train_data['Description'].tolist())
    test_encodings = tokenize_function(test_data['Description'].tolist())

    return train_encodings, train_data['Label'].tolist(), test_encodings, test_data['Label'].tolist()
