import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 加载训练和测试数据
    train_file_path = './TrainTest/train_dataset.csv'  # 根据需要更新文件路径
    test_file_path = './TrainTest/test_dataset.csv'    # 根据需要更新文件路径
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    # 准备数据
    X_train = train_data[['NLPscores', 'CVscores']]
    y_train = train_data['Label']
    X_test = test_data[['NLPscores', 'CVscores']]
    y_test = test_data['Label']

    # 训练随机森林模型
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # 在测试数据上进行预测（获取概率分数）
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # 计算准确率
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"准确度: {accuracy}")

    # 将概率分数保存回测试数据集
    test_data['RFscores'] = y_pred_proba
    output_file_path = './TrainTest/test_dataset.csv'
    test_data.to_csv(output_file_path, index=False)

    print("概率分数已保存至:", output_file_path)

if __name__ == "__main__":
    main()
