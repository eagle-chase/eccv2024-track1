python infer/infer.py \
    --input-path eval/data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl \
    --image-root eval/data/coda-lm/ \
    --model-path model/llava-llama-3-8b-v1_1-transformers \
    --save-path eval/result/pred \
    --task-name driving_suggestion \
    --device-id 0 

python infer/infer.py \
    --input-path eval/data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl \
    --image-root eval/data/coda-lm/ \
    --model-path model/llava-llama-3-8b-v1_1-transformers \
    --save-path eval/result/pred \
    --task-name general_perception \
    --device-id 0 

python infer/infer.py \
    --input-path eval/data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl \
    --image-root eval/data/coda-lm/ \
    --model-path model/llava-llama-3-8b-v1_1-transformers \
    --save-path eval/result/pred \
    --task-name region_perception \
    --device-id 0 
