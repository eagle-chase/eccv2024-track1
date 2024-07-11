# export PYTHONPATH="/home/luca.huang/.conda/envs/xtuner/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="/home/luca.huang/.conda/envs/xtuner/bin:$PATH"
NPROC_PER_NODE=4 python /home/luca.huang/.conda/envs/xtuner/bin/xtuner train \
xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune.py \
--deepspeed deepspeed_zero2
# xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py \  