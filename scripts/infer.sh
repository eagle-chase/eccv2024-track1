export PYTHONPATH="${PYTHONPATH}:/data/home/ouyangmt/eccv2024-track1"
python infer/inference.py \
    --llava_model_path model/llava-llama-3-8b-finetune \
    --llm_model_path model/Meta-Llama-3-8B-Instruct \
    --visual_encoder_path model/clip-vit-large-patch14-336 \
    --input-path /data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl \
    --image-root /data/home/ouyangmt/eccv2024-track1/data/coda-lm/ \
    --save-path /data/home/ouyangmt/eccv2024-track1/eval/result/pred \
    --task-name driving_suggestion \
    --device-id 1 \