```
python infer/infer.py \
    --input-path /data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Test/vqa_anno/driving_suggestion.jsonl \
    --image-root /data/home/ouyangmt/eccv2024-track1/data/coda-lm/ \
    --model-path model/llava-llama-3-8b-v1_1-transformers \
    --save-path /data/home/ouyangmt/eccv2024-track1/eval/result/pred \
    --task-name driving_suggestion \
    --device-id 1 \
```