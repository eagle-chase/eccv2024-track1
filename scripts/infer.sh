## set your project path
export PYTHONPATH="${PYTHONPATH}:/data/home/ouyangmt/eccv2024-track1"

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


# ======================= xtuner infer =======================

python infer/inference.py \
    --llava_model_path model/llava-llama-3-8b-finetune \
    --llm_model_path model/Meta-Llama-3-8B-Instruct \
    --visual_encoder_path model/clip-vit-large-patch14-336 \
    --input-path data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl \
    --image-root data/coda-lm/ \
    --save-path eval/result/pred \
    --task-name driving_suggestion \
    --model-type xtuner \

# python infer/inference.py \
#     --llava_model_path model/llava-llama-3-8b-finetune \
#     --llm_model_path model/Meta-Llama-3-8B-Instruct \
#     --visual_encoder_path model/clip-vit-large-patch14-336 \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --save-path eval/result/pred \
#     --task-name general_perception \
#     --model-type xtuner \

# python infer/inference.py \
#     --llava_model_path model/llava-llama-3-8b-finetune \
#     --llm_model_path model/Meta-Llama-3-8B-Instruct \
#     --visual_encoder_path model/clip-vit-large-patch14-336 \
#     --input-path data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl \
#     --image-root data/coda-lm/ \
#     --save-path eval/result/pred \
#     --task-name region_perception \
#     --model-type xtuner \
