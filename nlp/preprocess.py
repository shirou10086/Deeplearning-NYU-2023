'''
import pandas as pd
import os

# 指定包含 CSV 文件的文件夹路径
folder_path = './dataset'

# 找到文件夹中所有的 CSV 文件
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 合并所有 CSV 文件
all_data = pd.concat([pd.read_csv(file) for file in csv_files])


# 保存到新的 CSV 文件
merged_file_path = './merged_dataset.csv'
all_data.to_csv(merged_file_path, index=False)
print(f"Merged dataset saved to {merged_file_path}")
'''
import pandas as pd

# 加载CSV文件
file_path = 'editmergeddataset.csv'  # 替换成您的文件路径
data = pd.read_csv(file_path)

# 提取"D"列中文件名的第一个词，并保存到第六列（在Python中索引为5）
# 这里我们假设昆虫名称和文件编号之间有空格分隔，如 "ants (1).jpg"
data.iloc[:, 5] = data.iloc[:, 3].str.extract(r'(^[\w-]+)')

# 保存修改后的CSV文件
data.to_csv('editeditmergeddataset.csv', index=False)  # 您可以根据需要修改保存的文件名
