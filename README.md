# LLAVA

```
conda create --name xtuner python=3.10 -y
conda activate xtuner
pip install -U 'xtuner[deepspeed]'

mkdir model
cd model
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers
cd llava-llama-3-8b-v1_1-transformers
git lfs pull
```