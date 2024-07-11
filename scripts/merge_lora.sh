# FINETUNE_CFG='xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune.py'
# PTH_PATH='work_dirs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune/iter_5632.pth'
# SAVE_PATH='model/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune'

LLM_ADAPTER='model/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune/llm_adapter'
LLM='model/Meta-Llama-3-8B-Instruct'
SAVE_PATH='model/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e4_gpu8_finetune/merged_llm'

# (LLM) 
xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH

# (CLIP) 
# xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip