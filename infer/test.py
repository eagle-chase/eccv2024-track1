from common_utils import load_jsonl, save_jsonl
import argparse
import os
from tqdm import tqdm
task_to_prompt_file = {
    'driving_suggestion': 'driving_suggestion.txt',
    'general_perception': 'general_perception.txt',
    'region_perception': 'region_perception.txt'
}

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
batch_size = 4
input_jsonl_lst = [
    f'data/results/infer/Mini/driving_suggestion.jsonl',
    f'data/results/infer/Mini/general_perception.jsonl',
    f'data/results/infer/Mini/region_perception.jsonl',
]
for input_path in input_jsonl_lst:
    image_root = 'data/coda-lm/'
    question_list, image_path_list, origin_data = load_input(input_path, image_root)
    
    task_name = get_task_name(input_path)
    prompt_file_path = os.path.join('infer/', task_to_prompt_file[task_name])
with open(prompt_file_path, 'r') as file:
    task_prompt = file.read().strip()

question_batch = [question_list[i:i+batch_size] for i in range(0, len(question_list), batch_size)]
image_batch = [image_path_list[i:i+batch_size] for i in range(0, len(image_path_list), batch_size)]

infer_output = []
for questions, images in tqdm(zip(question_batch, image_batch)):
    prompts = [(task_prompt + ' ' + question, image) for question, image in zip(questions, images)]

print(prompts[0])