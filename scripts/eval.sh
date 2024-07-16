SPLIT='Mini-Val'
OPENAI_KEY='sk-xWQahTx28GvonjjC01F2C292581a4073A812D418Df90Dd2d'
API_BASE_URL='https://api.gpts.vin/v1'
MODEL_PATH='model/official/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune'

python eval/evaluation/stage1_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $MODEL_PATH/$SPLIT/general_perception.jsonl \
--save_path $MODEL_PATH/$SPLIT/general_perception \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL

python eval/evaluation/stage2_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $MODEL_PATH/$SPLIT/driving_suggestion.jsonl \
--save_path $MODEL_PATH/$SPLIT/driving_suggestion \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL

python eval/evaluation/convert2eval.py --reference_path $ROOT_TO_GT \
--prediction_path $MODEL_PATH/$SPLIT/region_perception.jsonl \

python eval/evaluation/stage3_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $MODEL_PATH/$SPLIT/region_perception_w_label.jsonl \
--save_path $MODEL_PATH/$SPLIT/region_perception \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL

python eval/merge_score.py --base_folder $MODEL_PATH/$SPLIT/ \