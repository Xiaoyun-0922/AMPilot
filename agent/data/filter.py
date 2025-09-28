import pandas as pd
import numpy as np

# --- 1. 请在这里配置您的文件信息 ---

# 文件一：包含抗菌肽 (AP) 的数据 (正样本)
FILE_AP = 'data/positive.csv' 
# 这个文件里代表序列的列名是什么？
COLUMN_AP_SEQUENCE = 'Sequence' 
# 这个文件里代表MIC的列名是什么？ (如果没有就填 None)
COLUMN_AP_MIC = None
# 这个文件里代表HC50的列名是什么？ (如果没有就填 None)
COLUMN_AP_HC50 = None 

# 文件二：包含非抗菌肽 (non-AP) 的数据 (负样本)
FILE_NON_AP = 'data/negative.csv'
# 这个文件里代表序列的列名是什么？
COLUMN_NON_AP_SEQUENCE = 'Sequence'
# 这个文件里代表MIC的列名是什么？ (如果没有就填 None)
COLUMN_NON_AP_MIC = None
# 这个文件里代表HC50的列名是什么？ (如果没有就填 None)
COLUMN_NON_AP_HC50 = None

# 输出的统一数据库文件名
OUTPUT_FILE = 'amp_database.csv'

# --- 2. 合并脚本逻辑 (通常无需修改) ---

def process_and_merge():
    """
    Loads, standardizes, and merges the two datasets.
    """
    try:
        df_ap = pd.read_csv(FILE_AP)
        print(f"成功读取 {FILE_AP}，包含 {len(df_ap)} 条数据。")
        # 添加 is_amp 标记
        df_ap['is_amp'] = True
    except FileNotFoundError:
        print(f"错误：找不到文件 {FILE_AP}。请检查路径和文件名。")
        return

    try:
        df_non_ap = pd.read_csv(FILE_NON_AP)
        print(f"成功读取 {FILE_NON_AP}，包含 {len(df_non_ap)} 条数据。")
        # 添加 is_amp 标记
        df_non_ap['is_amp'] = False
    except FileNotFoundError:
        print(f"错误：找不到文件 {FILE_NON_AP}。请检查路径和文件名。")
        return

    # --- 标准化列名 ---
    # 定义一个映射，key是旧列名，value是新列名
    rename_map_ap = {COLUMN_AP_SEQUENCE: 'sequence'}
    if COLUMN_AP_MIC: rename_map_ap[COLUMN_AP_MIC] = 'mic'
    if COLUMN_AP_HC50: rename_map_ap[COLUMN_AP_HC50] = 'hc50'

    rename_map_non_ap = {COLUMN_NON_AP_SEQUENCE: 'sequence'}
    if COLUMN_NON_AP_MIC: rename_map_non_ap[COLUMN_NON_AP_MIC] = 'mic'
    if COLUMN_NON_AP_HC50: rename_map_non_ap[COLUMN_NON_AP_HC50] = 'hc50'
    
    df_ap.rename(columns=rename_map_ap, inplace=True)
    df_non_ap.rename(columns=rename_map_non_ap, inplace=True)

    # --- 合并数据 ---
    # 使用 pd.concat 来合并两个 DataFrame
    # ignore_index=True 会重新生成索引
    combined_df = pd.concat([df_ap, df_non_ap], ignore_index=True)

    # --- 整理最终的列 ---
    # 确保 'sequence', 'mic', 'hc50', 'is_ap' 这几列都存在
    final_columns = ['sequence', 'mic', 'hc50', 'is_amp']
    for col in final_columns:
        if col not in combined_df.columns:
            # 如果某列不存在（例如原始文件都没有hc50），就创建它并填充空值
            combined_df[col] = np.nan 
    
    # 重新排列列的顺序，并丢弃其他不必要的列
    final_df = combined_df[final_columns]

    # --- 数据清洗：去除重复的序列 ---
    initial_rows = len(final_df)
    final_df.drop_duplicates(subset=['sequence'], keep='first', inplace=True)
    print(f"去除了 {initial_rows - len(final_df)} 个重复序列。")

    # --- 数据清洗：根据序列长度过滤 (12-18 aa) ---
    initial_rows_before_len_filter = len(final_df)
    final_df = final_df[final_df['sequence'].str.len().between(12, 18)]
    print(f"根据长度 (12-18 aa) 过滤，去除了 {initial_rows_before_len_filter - len(final_df)} 个序列。")

    # --- 保存结果 ---
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n合并完成！总共 {len(final_df)} 条数据已保存到 {OUTPUT_FILE}")
    print("最终数据库的列名:", final_df.columns.tolist())
    print("数据预览:\n", final_df.head())


if __name__ == '__main__':
    process_and_merge()
