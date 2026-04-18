import pandas as pd
import json

# 读取生成的 samples 文件 (替换为你的实际文件名)
file_path = 'Qwen__Qwen3.5-2B/samples_manufacture_qa_single_2026-04-10T09-09-04.981310.jsonl'
# file_path = 'Qwen__Qwen2.5-3B-Instruct/samples_manufacture_qa_multi_2026-04-04T21-15-41.146715.jsonl'
# file_path = '__mnt__nfs__huggingface__Qwen__Qwen3-4B-Instruct/samples_manufacture_qa_multi_2026-04-05T10-16-01.320803.jsonl'
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        # 提取元数据
        difficulty = item['doc']['difficulty']
        q_type = item['doc']['type']
        
        # 提取这道题的得分 (以 recall@1 和 ndcg@5 为例)
        # 注意：lm-eval 的 metrics 结果存在 item['resps'] 或直接在顶层，具体视版本而定
        # 最新版通常在 item['doc_id'] 同级的字典里，或者需要你自己对比 target 和 resps
        # 假设这里我们提取是否命中 (Hit@1 / recall@1)
        score = item['acc'] if 'acc' in item else item.get('recall@1', 0) 
        
        data.append({
            'difficulty': difficulty,
            'type': q_type,
            'score': score
        })

df = pd.DataFrame(data)

# 1. 按难度 (difficulty) 分组查看平均分
print("=== 按难度分组成绩 ===")
print(df.groupby('difficulty')['score'].mean())

# 2. 按类型 (type) 分组查看平均分
print("\n=== 按类型分组成绩 ===")
print(df.groupby('type')['score'].mean())