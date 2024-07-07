from PIL import Image
import os
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List

from utils import load_json, save_json, parse_args

def model_infer(
    question_list: List[str],
    image_path_list: List[str],
    model_path: str = "model/llava-llama-3-8b-v1_1-transformers",
    device_id: int = 0,
):
    assert len(question_list) == len(image_path_list), \
        "The number of problems does not match the number of images."
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device_id)
    processor = AutoProcessor.from_pretrained(model_path)
    
    infer_result = []
    for quesion, image_path in tqdm(zip(question_list, image_path_list)): 
        prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n<image>\n{quesion}<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n")
        raw_image = Image.open(image_path)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(device_id, torch.float16)
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        decode_output = processor.decode(output[0][2:], skip_special_tokens=True)
        infer_result.append(decode_output)
    
    return infer_result


def load_input(input_path: str, image_root: str):
    data = load_json(input_path)
    question_list, image_path_list = [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
    return question_list, image_path_list, data


def main():
    args = parse_args()

    question_list, image_path_list, origin_data = load_input(args.input_path, args.image_root)

    origin_data = origin_data
    
    infer_output = model_infer(
        question_list=question_list,
        image_path_list=image_path_list,
        model_path=args.model_path,
        device_id=args.device_id,
    )
    assert len(infer_output) == len(origin_data), \
        "The number of results is not equal to the number of original data."
    for idx in range(len(infer_output)):
        ans = infer_output[idx].split("assistant\n")[1]
        origin_data[idx]["answer"] = ans 
    
    save_json(os.path.join(args.save_path, args.task_name + ".jsonl"), origin_data)


if __name__ == "__main__":
    main()
