import json
import argparse


def load_json(path: str):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def save_json(path: str, data: any):
    with open(path, 'w') as file:
        for obj in data:
            line = json.dumps(obj)
            file.write(line + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    
    args, argv = parser.parse_known_args()

    return args
