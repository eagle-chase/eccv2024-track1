from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
from common_utils import load_jsonl, save_jsonl
import argparse
import os
from tqdm import tqdm

def load_input(input_path: str, image_root: str):
    data = load_jsonl(input_path)
    question_list, image_path_list = [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
    return question_list, image_path_list, data

def main():

    batch_size = 4
    pipe = pipeline('model/official/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune',
                    chat_template_config=ChatTemplateConfig(model_name='llama3'))

    input_jsonl_lst = [
        'data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl',
        'data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl',
        'data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl',
    ]
    for input_path in input_jsonl_lst:
        image_root = 'data/coda-lm/'
        question_list, image_path_list, origin_data = load_input(input_path, image_root)

        question_batch = [question_list[i:i+batch_size] for i in range(0, len(question_list), batch_size)]
        image_batch = [image_path_list[i:i+batch_size] for i in range(0, len(image_path_list), batch_size)]

        infer_output = []
        for questions, images in tqdm(zip(question_batch, image_batch)):
            prompts = [(question, load_image(image)) for question, image in zip(questions, images)]
            response = pipe(prompts)
            infer_text= [r.text for r in response]
            infer_output.extend(infer_text)

        for idx in range(len(infer_output)):
            origin_data[idx]["answer"] = infer_output[idx]
        
        save_jsonl(os.path.join('data/results/infer', input_path.split('/')[-1]), origin_data)


if __name__ == '__main__':
    main()