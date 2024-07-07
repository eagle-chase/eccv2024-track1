OPENAI_KEY='sk-xWQahTx28GvonjjC01F2C292581a4073A812D418Df90Dd2d'
API_BASE_URL='https://api.gpts.vin/v1'
ROOT_TO_RESULTS='eval/result/pred'
ROOT_TO_GT='eval/result/GT'

python eval/evaluation/stage1_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $ROOT_TO_RESULTS/general_perception.jsonl \
--save_path eval/result/score/general_perception \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL

python eval/evaluation/stage2_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $ROOT_TO_RESULTS/driving_suggestion.jsonl \
--save_path eval/esult/score/driving_suggestion \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL
   
python eval/evaluation/convert2eval.py --reference_path $ROOT_TO_GT \
--prediction_path $ROOT_TO_RESULTS/region_perception.jsonl \

python eval/evaluation/stage3_eval_batch.py --reference_path $ROOT_TO_GT \
--prediction_path $ROOT_TO_RESULTS/region_perception_w_label.jsonl \
--save_path eval/esult/score/region_perception \
--model_name gpt-4o-2024-05-13 \
--api_key $OPENAI_KEY \
--api_base_url $API_BASE_URL
