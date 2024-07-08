from PIL import Image
import os
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List
from common_utils import load_jsonl, save_jsonl
from infer.utils import parse_args
import os.path as osp
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer, 
    AutoProcessor, LlavaForConditionalGeneration,
    CLIPImageProcessor, CLIPVisionModel, GenerationConfig
)
from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    PROMPT_TEMPLATE, SYSTEM_TEMPLATE
)
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def xtuner_infer(
    question_list: str,
    image_path_list: str,
    llm_model_path: str,
    llava_model_path: str,
    visual_encoder_path: str,
):
    torch_dtype = "fp16"
    visual_select_layer = -2
    prompt_template = "llama3_chat"
    max_new_tokens = 2048
    temperature = 0.1
    top_k = 40
    top_p = 0.75
    repetition_penalty = 1.0
    model_kwargs = {
        'quantization_config': None,
        'load_in_8bit': False,
        'device_map': 'auto',
        'offload_folder': None,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[torch_dtype]
    }

    llm = AutoModelForCausalLM.from_pretrained(llm_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_path,
        trust_remote_code=True,
        encode_special_tokens=True
    )

    llava_path = snapshot_download(
        repo_id=llava_model_path) if not osp.isdir(
            llava_model_path) else llava_model_path

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert visual_encoder_path is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert visual_encoder_path is not None, (
            'Please specify the `--visual-encoder`!')
        # visual_encoder_path = visual_encoder_path
    visual_encoder = CLIPVisionModel.from_pretrained(
        visual_encoder_path,
        torch_dtype=TORCH_DTYPE_MAP[torch_dtype])
    image_processor = CLIPImageProcessor.from_pretrained(
        visual_encoder_path)
    print(f'Load visual_encoder from {visual_encoder_path}')

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')
        llm = PeftModel.from_pretrained(
            llm,
            adapter_path,
            offload_folder=None,
            trust_remote_code=True)
        print(f'Load LLM adapter from {llava_model_path}')
    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder,
            adapter_path,
            offload_folder=None)
        print(f'Load visual_encoder adapter from {llava_model_path}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    projector = AutoModel.from_pretrained(
        projector_path,
        torch_dtype=TORCH_DTYPE_MAP[torch_dtype],
        trust_remote_code=True)
    print(f'Load projector from {llava_model_path}')

    projector.cuda()
    projector.eval()
    visual_encoder.cuda()
    visual_encoder.eval()

    llm.eval()
    infer_result = []
    for image_path, text_input in tqdm(zip(image_path_list, question_list)):
        image = load_image(image_path)
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        pixel_values = projector(
            visual_outputs.hidden_states[visual_select_layer][:, 1:])

        stop_words = []
        if prompt_template:
            template = PROMPT_TEMPLATE[prompt_template]
            stop_words += template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=tokenizer, stop_words=stop_words)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        inputs = ""
        text = text_input
        if image_path is not None:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[prompt_template]

            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name="BOT")
        else:
            prompt_text = text
        inputs += prompt_text
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer.encode(chunk)
            else:
                cur_encode = tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)
        if len(generate_output[0]) >= max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {max_new_tokens}.')
        output_text = tokenizer.decode(generate_output[0][:-1])
        infer_result.append(output_text)
    
    return infer_result

def huggingface_infer(
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
        infer_result.append(decode_output.split("assistant\n")[1])
    
    return infer_result


def load_input(input_path: str, image_root: str):
    data = load_jsonl(input_path)
    question_list, image_path_list = [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
    return question_list, image_path_list, data


def main():
    args = parse_args()

    question_list, image_path_list, origin_data = load_input(args.input_path, args.image_root)

    origin_data = origin_data

    if args.model_type == "huggingface":
        infer_output = huggingface_infer(
            question_list=question_list,
            image_path_list=image_path_list,
            model_path=args.model_path,
            device_id=args.device_id,
        )
    else:
        infer_output = xtuner_infer(
            question_list=question_list,
            image_path_list=image_path_list,
            llava_model_path=args.llava_model_path,
            llm_model_path=args.llm_model_path,
            visual_encoder_path=args.visual_encoder_path,
        )

    assert len(infer_output) == len(origin_data), \
        "The number of results is not equal to the number of original data."
    for idx in range(len(infer_output)):
        origin_data[idx]["answer"] = infer_output[idx] 
    
    save_jsonl(os.path.join(args.save_path, args.task_name + ".jsonl"), origin_data)


if __name__ == "__main__":
    main()
