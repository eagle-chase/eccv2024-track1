import streamlit as st
from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
from PIL import Image
from common_utils import load_jsonl
import os

# 加载模型
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline('model/official/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune',
                    chat_template_config=ChatTemplateConfig(model_name='llama3'))

def load_input(input_path: str, image_root: str):
    data = load_jsonl(input_path)
    question_list, image_path_list = [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
    return question_list, image_path_list, data

def main():
    st.title("Model Inference Visualization")
    model = load_model()

    sample_input_path = 'data/coda-lm/CODA-LM/Val/vqa_anno/driving_suggestion.jsonl'
    input_path = st.text_input("Please enter your input file path", sample_input_path)
    image_root = st.text_input("Please enter your image root path", 'data/coda-lm/')

    question_list, image_path_list, _ = load_input(input_path, image_root)

    selected_index = st.selectbox("Select a question and image", range(len(question_list)))

    # if st.button("Infer"):
    if input_path and image_root:
        if selected_index < len(question_list):
            question = question_list[selected_index]
            image_path = image_path_list[selected_index]
            image = Image.open(image_path)
            st.image(image, caption='Input Image.', use_column_width=True)
            st.write("Question: ", question)
            response = model([(question, load_image(image_path))])
            infer_text = [r.text for r in response]
            st.write("Inference Result: ", infer_text[0])
        else:
            st.write("No more images to infer.")
    else:
        st.write("Please input both input file path and image root path.")


if __name__ == "__main__":
    main()