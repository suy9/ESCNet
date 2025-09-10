# ESCNet

This repository is the official implementation of ESCNet:Edge-Semantic Collaborative Network for Camouflaged Object Detection


### 安装建议

#### 1. **使用 pip 安装**

直接运行以下命令：

```bash
pip install -r requirements.txt
```

#### 2. **使用 conda 安装**

对于某些包（如 `torch`, `cudatoolkit`），建议优先使用 conda 安装以获得更好的兼容性：

```bash
conda create -n myenv python=3.11.11
conda activate myenv

# 安装 PyTorch 和相关库（推荐 conda）
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```
