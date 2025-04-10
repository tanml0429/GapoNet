

# 安装mamba依赖
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal_conv1d==1.1.1
pip install mamba-ssm==1.2.0.post1
git clone https://github.com/hustvl/Vim.git
# copy mamba-ssm dir in vim to conda env site-package dir
cp -rf mamba-1p1p1/mamba_ssm /opt/miniconda3/envs/mamba/lib/python3.10/site-packages
```
