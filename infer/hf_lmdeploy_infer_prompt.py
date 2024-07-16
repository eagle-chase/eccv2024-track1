from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
from common_utils import load_jsonl, save_jsonl
import argparse
import os
from tqdm import tqdm

task_to_prompt_file = {
    'driving_suggestion': 'driving_suggestion.txt',
    'general_perception': 'general_perception.txt',
    'region_perception': 'region_perception.txt'
}

# 获取当前任务名的函数
def get_task_name(input_path):
    for task_name in task_to_prompt_file.keys():
        if task_name in input_path:
            return task_name
    return None

def load_input(input_path: str, image_root: str):
    data = load_jsonl(input_path)
    question_list, image_path_list = [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
    return question_list, image_path_list, data

def main():

    batch_size = 3
    model_path = 'model/official/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune'
    pipe = pipeline(model_path,
                    chat_template_config=ChatTemplateConfig(model_name='llama3'))

    split = 'Mini-Val'
    save_folder = 'Mini-Val_prompt'
    input_jsonl_lst = [
        f'data/coda-lm/CODA-LM/{split}/vqa_anno/driving_suggestion.jsonl',
        f'data/coda-lm/CODA-LM/{split}/vqa_anno/general_perception.jsonl',
        f'data/coda-lm/CODA-LM/{split}/vqa_anno/region_perception.jsonl',
    ]
    for input_path in input_jsonl_lst:
        image_root = 'data/coda-lm/'
        question_list, image_path_list, origin_data = load_input(input_path, image_root)
       
        task_name = get_task_name(input_path)
        
        prompt_file_path = os.path.join('infer/template', task_to_prompt_file[task_name])
        with open(prompt_file_path, 'r') as file:
            task_prompt = file.read().strip()

        question_batch = [question_list[i:i+batch_size] for i in range(0, len(question_list), batch_size)]
        image_batch = [image_path_list[i:i+batch_size] for i in range(0, len(image_path_list), batch_size)]

        infer_output = []
        for questions, images in tqdm(zip(question_batch, image_batch)):
            prompts = [(task_prompt + ' ' + question, load_image(image)) for question, image in zip(questions, images)]
            response = pipe(prompts)
            infer_text= [r.text for r in response]
            infer_output.extend(infer_text)

        for idx in range(len(infer_output)):
            origin_data[idx]["answer"] = infer_output[idx]
        
        os.mkdirs(os.path.join(model_path, 'results'), exist_ok=True)
        if not os.path.exists(os.path.join(model_path, 'results', save_folder)):
            os.makedirs(os.path.join(model_path, 'results', save_folder))
        save_jsonl(os.path.join(model_path, 'results', save_folder, input_path.split('/')[-1]), origin_data)


if __name__ == '__main__':
    main()