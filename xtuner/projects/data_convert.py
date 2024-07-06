from common_utils import load_jsonl, save_json
import time
# export PYTHONPATH="${PYTHONPATH}:/data/home/ouyangmt/eccv2024-track1"

def coda_lm2llava(anno_path: str, output_path: str):
    anno = load_jsonl(anno_path)
    llava_data = []
    for item in anno:
        new = {
            "id": round(time.time()*1000),
            "image": item["image"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{item['question']}",
                },
                {
                    "from": "gpt",
                    "value": item["answer"],
                }, 
            ]
        }
        llava_data.append(new)

    save_json(output_path, llava_data)
    


if __name__ == "__main__":
    anno_path_list = [
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/driving_suggestion.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/general_perception.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/region_perception.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/driving_suggestion.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/general_perception.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/region_perception.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/general_perception.jsonl",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/region_perception.jsonl",
    ]
    output_path_list = [
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/driving_suggestion_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/general_perception_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Mini/vqa_anno/region_perception_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/driving_suggestion_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/general_perception_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Train/vqa_anno/region_perception_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/general_perception_llava.json",
        "/data/home/ouyangmt/eccv2024-track1/data/coda-lm/CODA-LM/Val/vqa_anno/region_perception_llava.json",
    ]

    for anno_path, output_path in zip(anno_path_list, output_path_list):
        coda_lm2llava(anno_path, output_path)


    
