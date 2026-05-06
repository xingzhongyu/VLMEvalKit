#!/bin/bash

# 遇到错误时立即退出
set -e

# 定义需要运行的 Q 编号列表
Q_LIST=(4 3 1 5 2)

# 你的主评测脚本名称
MAIN_SCRIPT="run_benchmark_all.sh"

# 检查主脚本是否存在
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ 错误: 找不到主脚本 $MAIN_SCRIPT，请确保它和本脚本在同一目录下。"
    exit 1
fi

echo "================================================="
echo "🚀 开始批量执行评测任务: Q1, Q2, Q3, Q4, Q5"
echo "================================================="

# 遍历列表并依次执行
for q in "${Q_LIST[@]}"; do
    echo ""
    echo "▶️ 正在启动 Q${q} 的评测任务..."
    echo "执行命令: bash $MAIN_SCRIPT $q"
    
    # 执行主脚本并传入参数
    bash "$MAIN_SCRIPT" "$q"
    
    echo "✅ Q${q} 评测任务执行完成！"
    echo "-------------------------------------------------"
    
    # 可选：在两个任务之间暂停 10 秒，让系统/显存有时间完全释放和回收
    sleep 10
done

echo "🎉 所有评测任务 (Q1, Q2, Q3, Q5) 均已顺利完成！"