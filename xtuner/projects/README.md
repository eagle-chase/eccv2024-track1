1. 准备模型
- llama
- visual encoder
- llava pth

2. 准备数据
xtuner/projects/data_convert.py # 修改路径
把三个任务的json手动合并成一个 CODA-LM/Train/vqa_anno/all_llava.json

3. 修改配置文件
xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py
主要是batch size和模型、数据集路径

4. 8卡训练
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" NPROC_PER_NODE=8 xtuner train xtuner/projects/llava_llama3/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2