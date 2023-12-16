import pandas as pd
from sklearn.linear_model import LinearRegression
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
    y_test = test_data['Label']  # 确保测试集中也有标签列

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 在测试数据上进行预测
    y_pred_continuous = model.predict(X_test)

    # 计算准确率（根据阈值0.5将连续预测转换为二元类别）
    y_pred_binary = (y_pred_continuous > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"准确度: {accuracy}")

    # （可选）将连续预测结果保存回测试数据集
    test_data['IRscores'] = y_pred_continuous
    output_file_path = './TrainTest/test_dataset.csv'
    test_data.to_csv(output_file_path, index=False)

    print("连续预测结果已保存至:", output_file_path)

if __name__ == "__main__":
    main()
