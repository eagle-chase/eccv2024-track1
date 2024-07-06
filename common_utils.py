import json


def load_jsonl(path: str):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def save_jsonl(path: str, data: any):
    with open(path, 'w') as file:
        for obj in data:
            line = json.dumps(obj)
            file.write(line + '\n')

def load_json(path: str):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_json(path: str, data: any):
    with open(path, 'w') as file:
        json.dump(data, file)

