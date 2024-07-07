import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava_model_path", type=str, required=True)
    parser.add_argument("--llm_model_path", type=str, required=True)
    parser.add_argument("--visual_encoder_path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    
    args, argv = parser.parse_known_args()

    return args
