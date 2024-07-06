SYSTEM = ''
accumulative_counts = 128
batch_size = 1
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
    dict(
        evaluation_images='..data/coda-lm/test/images/0001.jpg',
        evaluation_inputs=[
            'Please describe this picture',
        ],
        every_n_iters=50000,
        image_processor=dict(
            pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
            trust_remote_code=True,
            type='transformers.CLIPImageProcessor.from_pretrained'),
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.llama3_chat',
        system='',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.EvaluateChatHook'),
]
data_path = '../data/coda-lmCODA-LM/Mini/vqa_anno/driving_suggestion_llava.json'
data_root = '../data/coda-lm'
dataloader_num_workers = 0
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=50000,
        max_keep_ckpts=2,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 50000
evaluation_images = '..data/coda-lm/test/images/0001.jpg'
evaluation_inputs = [
    'Please describe this picture',
]
image_folder = '../data/coda-lmCODA-LM/'
image_processor = dict(
    pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
    trust_remote_code=True,
    type='transformers.CLIPImageProcessor.from_pretrained')
launcher = 'none'
llava_dataset = dict(
    data_path=
    '../data/coda-lmCODA-LM/Mini/vqa_anno/driving_suggestion_llava.json',
    dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
    image_folder='../data/coda-lmCODA-LM/',
    image_processor=dict(
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
        trust_remote_code=True,
        type='transformers.CLIPImageProcessor.from_pretrained'),
    max_length=1472,
    pad_image_to_square=True,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.llama3_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.LLaVADataset')
llm_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 0.0002
max_epochs = 1
max_length = 1472
max_norm = 1
model = dict(
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm=dict(
        pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
        quantization_config=dict(
            bnb_4bit_compute_dtype='torch.float16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            load_in_4bit=True,
            load_in_8bit=False,
            type='transformers.BitsAndBytesConfig'),
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    llm_lora=dict(
        bias='none',
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    pretrained_pth='./projects/llava_llama3/ckpts/iter_2181_new.pth',
    type='xtuner.model.LLaVAModel',
    visual_encoder=dict(
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
        type='transformers.CLIPVisionModel.from_pretrained'))
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0002,
        type='torch.optim.AdamW',
        weight_decay=0),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.03,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=0.03,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
pretrained_pth = './projects/llava_llama3/ckpts/iter_2181_new.pth'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.llama3_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
runner_type = 'FlexibleRunner'
save_steps = 50000
save_total_limit = 2
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False, initial_scale_power=16),
        gradient_accumulation_steps='auto',
        gradient_clipping='auto',
        train_micro_batch_size_per_gpu='auto',
        zero_allow_untested_optimizer=True,
        zero_force_ds_cpu_optimizer=False,
        zero_optimization=dict(overlap_comm=True, stage=2)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=128,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=1,
    type='xtuner.engine.DeepSpeedStrategy')
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=1, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_path=
        '../data/coda-lmCODA-LM/Mini/vqa_anno/driving_suggestion_llava.json',
        dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
        image_folder='../data/coda-lmCODA-LM/',
        image_processor=dict(
            pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
            trust_remote_code=True,
            type='transformers.CLIPImageProcessor.from_pretrained'),
        max_length=1472,
        pad_image_to_square=True,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.llama3_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.LLaVADataset'),
    num_workers=0,
    sampler=dict(
        length_property='modality_length',
        per_device_batch_size=128,
        type='xtuner.dataset.samplers.LengthGroupedSampler'))
visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
visualizer = None
warmup_ratio = 0.03
weight_decay = 0
work_dir = './work_dirs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu1_finetune_copy'
