CONFIG_NAME='llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune'
FINETUNE_CFG="xtuner/projects/llava_llama3/configs/${CONFIG_NAME}.py"
PTH_PATH="work_dirs/${CONFIG_NAME}/iter_4104.pth"
SAVE_PATH="model/official/${CONFIG_NAME}"

xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH \
--save-format official