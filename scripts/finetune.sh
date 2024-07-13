CUDA_VISIBLE_DEVICES="3,4" NPROC_PER_NODE=2 xtuner train \
xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune.py \
--deepspeed deepspeed_zero2
