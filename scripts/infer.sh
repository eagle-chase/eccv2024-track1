## set your project path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
LLM_PATH='model/Meta-Llama-3-8B-Instruct'
VISUAL_ENCODER_PATH='model/clip-vit-large-patch14-336'
LLAVA_PATH='model/xtuner/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune'

# ======================= xtuner infer =======================

python infer/inference.py \
    --llava_model_path $LLAVA_PATH \
    --llm_model_path $LLM_PATH \
    --input-path data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl \
    --image-root data/coda-lm/ \
    --save-path eval/result/pred \
    --task-name driving_suggestion \
    --model-type xtuner \
    --visual_encoder_path $VISUAL_ENCODER_PATH \

# python infer/inference.py \
#     --llava_model_path $LLAVA_PATH \
#     --llm_model_path $LLM_PATH \
#     --visual_encoder_path $VISUAL_ENCODER_PATH \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --save-path eval/result/pred \
#     --task-name general_perception \
#     --model-type xtuner \

# python infer/inference.py \
#     --llava_model_path $LLAVA_PATH \
#     --llm_model_path $LLM_PATH \
#     --visual_encoder_path $VISUAL_ENCODER_PATH \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --save-path eval/result/pred \
#     --task-name region_perception \
#     --model-type xtuner \


# ======================= huggingface infer =======================

# python infer/inference.py \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl \
#     --image-root data/coda-lm/ \
#     --model-path model/llava-llama-3-8b-v1_1-transformers \
#     --save-path eval/result/pred \
#     --task-name driving_suggestion \
#     --device-id 0 \
#     --model-type huggingface \

# python infer/inference.py \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --model-path model/llava-llama-3-8b-v1_1-transformers \
#     --save-path eval/result/pred \
#     --task-name general_perception \
#     --device-id 0 \
#     --model_type huggingface \

# python infer/inference.py \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --model-path model/llava-llama-3-8b-v1_1-transformers \
#     --save-path eval/result/pred \
#     --task-name region_perception \
#     --device-id 0 \
#     --model-type huggingface \

