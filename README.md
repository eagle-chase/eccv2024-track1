# LLAVA

benchmark_finetune_config = xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune.py

环境配置
```
conda create --name xtuner python=3.10 -y
conda activate xtuner
cd xtuner
pip install -e '.[deepspeed]'

pip install lmdeploy==0.4.2
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
```

官方llava模型（未微调）
```
mkdir model
cd model
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers
cd llava-llama-3-8b-v1_1-transformers
git lfs pull
```

训练
```
source env.sh
bash scripts/finetune.sh
```

模型转换（训练模型 -> offical格式）
```
bash scripts/convert_pth_to_official.sh
```



离线推理及web推理（基于lmdeploy
```
python infer/hf_lmdeploy_infer.py

lmdeploy serve gradio --chat-template template/llama3_chat_template.json --model-name llava-v1 model/official/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune
```
