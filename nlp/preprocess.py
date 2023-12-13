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
