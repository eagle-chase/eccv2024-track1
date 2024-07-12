CONIFG='llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune'
# lmdeploy lite smooth_quant model/official/$CONIFG \
# --work-dir model/8_bit_quant/$CONIFG \


lmdeploy lite auto_awq \
  model/official/$CONIFG \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir model/4_bit_quant/$CONIFG \
  --device cpu
