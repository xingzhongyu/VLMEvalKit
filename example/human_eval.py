import pandas as pd

def create_evaluation_excel(input_csv_path, output_excel_path):
    # 1. 读取原始 CSV 数据
    # 注意：如果您的原始数据有特殊的编码（如 gbk），请添加 encoding='gbk' 参数
    df = pd.read_csv(input_csv_path, sep='\t') # 根据您提供的数据示例，似乎是制表符或逗号分隔，请根据实际情况修改 sep=',' 或 '\t'
    
    # 2. 定义新增的评估列
    eval_columns = [
        'Score: Domain Accuracy (1-5)',
        'Score: Visual Answerability (1-5)',
        'Score: Distractor Quality (1-5)',
        'Score: Question Clarity (1-5)',
        'Score: Justification Quality (1-5)',
        'Expert Comments / Suggested Fixes'
    ]
    
    # 将这些列添加到 DataFrame 中，初始值为空
    for col in eval_columns:
        df[col] = ''
        
    # 3. 重新排列列的顺序，把最重要的放在前面，元数据放到最后
    base_cols = ['index', 'image_path', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'answer', 'justification']
    meta_cols = ['category', 'type', 'subtype', 'difficulty', 'manufacturer', 'material_capabilities', 'process_capabilities', 'answer_source']
    
    # 确保所有列都在数据中
    existing_base = [col for col in base_cols if col in df.columns]
    existing_meta = [col for col in meta_cols if col in df.columns]
    
    final_columns = existing_base + eval_columns + existing_meta
    df = df[final_columns]
    
    # 4. 使用 xlsxwriter 导出并格式化 Excel
    writer = pd.ExcelWriter(output_excel_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='VQA_Evaluation')
    
    # 获取 workbook 和 worksheet 对象以进行高级格式设置
    workbook  = writer.book
    worksheet = writer.sheets['VQA_Evaluation']
    
    # 定义格式
    wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#D7E4BC',
        'border': 1
    })
    
    # 写入带格式的表头
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    
    # 设置列宽
    worksheet.set_column('A:A', 8)  # index
    worksheet.set_column('B:B', 35, wrap_format) # image_path
    worksheet.set_column('C:C', 40, wrap_format) # question
    worksheet.set_column('D:K', 15) # options A-H
    worksheet.set_column('L:L', 10) # answer
    worksheet.set_column('M:M', 50, wrap_format) # justification
    worksheet.set_column('N:R', 15) # Score columns
    worksheet.set_column('S:S', 40, wrap_format) # Comments
    worksheet.set_column('T:Z', 15) # Metadata
    
    # 冻结首行和前三列 (index, image_path, question)
    worksheet.freeze_panes(1, 3)
    
    # 为打分列 (N列 到 R列，即索引 13 到 17) 添加 1-5 的数据验证下拉菜单
    # 注意：这里的索引可能因为实际列数有所偏移，我们动态找到这些列的索引
    score_col_indices = [df.columns.get_loc(col) for col in eval_columns[:-1]] # 不包含 Comment 列
    
    for col_idx in score_col_indices:
        col_letter = chr(ord('A') + col_idx) if col_idx < 26 else chr(ord('A') + (col_idx // 26) - 1) + chr(ord('A') + (col_idx % 26))
        # 应用于所有数据行 (假设最多 10000 行)
        cell_range = f'{col_letter}2:{col_letter}10000'
        
        worksheet.data_validation(cell_range, {
            'validate': 'list',
            'source': [1, 2, 3, 4, 5],
            'input_message': 'Please select a score from 1 to 5',
            'error_message': 'Score must be an integer between 1 and 5'
        })
        
    writer.close()
    print(f"✅ 评估文件已成功生成：{output_excel_path}")

# ================= 使用示例 =================
# 假设您的原始数据文件名为 'raw_data.csv'
# 想要生成的评估文件名为 'expert_evaluation_task.xlsx'

if __name__ == "__main__":
    # 请将这里的路径替换为您实际的文件路径
    input_file = 'raw_data.csv' 
    output_file = 'expert_evaluation_task.xlsx'
    
    try:
        create_evaluation_excel(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {input_file}，请检查路径是否正确，或将您的示例保存为该文件名。")