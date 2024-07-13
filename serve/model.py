import os
import torch
import os.path as osp
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer, 
    CLIPImageProcessor, CLIPVisionModel, GenerationConfig
)
from xtuner.dataset.utils import expand2square
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    PROMPT_TEMPLATE, SYSTEM_TEMPLATE
)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

class LLaVA():
    def __init__(
        self,
        llm_model_path: str,
        llava_model_path: str,
        visual_encoder_path: str = None,
        prompt_template: str = "llama3_chat",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.75,
        repetition_penalty: float = 1.0,
    ):
        self.llm_model_path = llm_model_path
        self.visual_encoder_path = visual_encoder_path
        self.llava_model_path = llava_model_path
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty =repetition_penalty
        self.visual_select_layer = -2

        torch_dtype = "fp16"
        model_kwargs = {
            'quantization_config': None,
            'load_in_8bit': False,
            'device_map': 'auto',
            'offload_folder': None,
            'trust_remote_code': True,
            'torch_dtype': TORCH_DTYPE_MAP[torch_dtype]
        }
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_model_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_path,
            trust_remote_code=True,
            encode_special_tokens=True
        )

        llava_path = snapshot_download(
            repo_id=self.llava_model_path) if not osp.isdir(
                self.llava_model_path) else self.llava_model_path

        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert self.visual_encoder_path is None, (
                "Please don't specify the `--visual-encoder` since passed "
                '`--llava` contains a visual encoder!')
            self.visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert self.visual_encoder_path is not None, (
                'Please specify the `--visual-encoder`!')
            # visual_encoder_path = visual_encoder_path
        self.visual_encoder = CLIPVisionModel.from_pretrained(
            self.visual_encoder_path,
            torch_dtype=TORCH_DTYPE_MAP[torch_dtype])
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.visual_encoder_path)
        print(f'Load visual_encoder from {self.visual_encoder_path}')

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            self.llm = PeftModel.from_pretrained(
                self.llm,
                adapter_path,
                offload_folder=None,
                trust_remote_code=True)
            print(f'Load LLM adapter from {self.llava_model_path}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            self.visual_encoder = PeftModel.from_pretrained(
                self.visual_encoder,
                adapter_path,
                offload_folder=None)
            print(f'Load visual_encoder adapter from {self.llava_model_path}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        self.projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=TORCH_DTYPE_MAP[torch_dtype],
            trust_remote_code=True)
        print(f'Load projector from {self.llava_model_path}')

        self.projector.cuda()
        self.projector.eval()
        self.visual_encoder.cuda()
        self.visual_encoder.eval()

        self.llm.eval()


        stop_words = []
        if self.prompt_template:
            template = PROMPT_TEMPLATE[self.prompt_template]
            stop_words += template.get('STOP_WORDS', [])
        self.stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)

        self.gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )
        if self.prompt_template:
            self.template = PROMPT_TEMPLATE[self.prompt_template]

    def __call__(self, image, text):
        image = expand2square(
            image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(self.visual_encoder.dtype)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        if image is not None:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
            prompt_text = self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name="BOT")
        else:
            prompt_text = text

        chunk_encode = []
        for idx, chunk in enumerate(prompt_text.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(
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
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)
        if len(generate_output[0]) >= self.max_new_tokens:
            print(
                'Remove the memory of history responses, since '
                f'it exceeds the length limitation {self.max_new_tokens}.')
        output_text = self.tokenizer.decode(generate_output[0][:-1])
        
        return output_text

