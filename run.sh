#!/bin/bash

# 默认执行训练、测试和评估
TRAIN=true

# 检查是否有 --notrain 参数
if [[ "$1" == "--notrain" ]]; then
    TRAIN=false
fi

# 训练部分
if [ "$TRAIN" = true ]; then
    echo "开始训练..."
    torchrun --nproc_per_node=4 train.py --config config.yaml
fi

# 测试部分
echo "执行测试脚本..."
python test.py

# 评估部分
echo "执行评估脚本..."
python eval.py