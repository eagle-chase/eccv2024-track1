import streamlit as st
from PIL import Image
from serve.model import LLaVA
from serve.eval_model import EvalModel1, EvalModel2, EvalModel3
from common_utils import load_jsonl
import os
from xtuner.dataset.utils import load_image

@st.cache(allow_output_mutation=True)
def load_model():
    model = LLaVA(
        llm_model_path="model/Meta-Llama-3-8B-Instruct",
        visual_encoder_path="model/clip-vit-large-patch14-336",
        llava_model_path="model/xtuner/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e4_gpu8_finetune",
    )
    return model

@st.cache(allow_output_mutation=True)
def load_eval_model(stage):
    if stage == 1: return EvalModel1()
    elif stage == 2: return EvalModel2()
    elif stage == 3: return EvalModel3()
    else: raise NotImplementedError(f"Not Implemented for {stage}")

def load_input(input_path: str, image_root: str):
    data = load_jsonl(input_path)
    question_list, image_path_list, answer_list = [], [], []
    for sample in data:
        question_list.append(sample["question"])
        image_path_list.append(os.path.join(image_root, sample["image"]))
        answer_list.append(sample.get("answer", None))
    return question_list, image_path_list, answer_list, data

def main():
    st.title("Model Inference Visualization")
    model = load_model()
    task_eval_model = {
        "general_perception": load_eval_model(stage=1),
        "driving_suggestion": load_eval_model(stage=2),
        "region_perception": load_eval_model(stage=3),
    }

    input_file_root = 'data/coda-lm/CODA-LM/Val/vqa_anno/'
    task_name = st.selectbox("Select a task", task_eval_model.keys())
    input_path = os.path.join(input_file_root, task_name + ".jsonl")
    image_root = st.text_input("Please enter your image root path", 'data/coda-lm/')

    question_list, image_path_list, answer_list, _ = load_input(input_path, image_root)
    image_path = st.selectbox("Select a question and image", image_path_list)
    selected_index = image_path_list.index(image_path)

    # if st.button("Infer"):
    if input_path and image_root:
        question = question_list[selected_index]
        image_path = image_path_list[selected_index]
        answer = answer_list[selected_index]
        image = load_image(image_path)
        st.image(image, caption='Input Image.', use_column_width=True)
        question_text = f"""
            <div style='color:blue;'>
                Question: <br>{question}
            </div>
        """
        st.markdown(question_text, unsafe_allow_html=True)
        response = model(image, question)
        infer_result_text = f"""
            <div style='color:red;'>
                Inference Result: <br>{response}
            </div>
        """
        st.markdown(infer_result_text, unsafe_allow_html=True)
        
        if answer:
            answer_text = f"""
                <div style='color:green;'>
                    Answer: <br>{answer}
                </div>
            """
            st.markdown(answer_text, unsafe_allow_html=True)
            eval_result, score = task_eval_model[task_name](response, answer)
            
            eval_result_text = f"""
                <div style='color:blue;'>
                    Eval Result: <br>{eval_result}
                </div>
            """
            st.markdown(eval_result_text, unsafe_allow_html=True)

            score_text = f"""
                <div style='color:blue;'>
                    Score: {score}
                </div>
            """
            st.markdown(score_text, unsafe_allow_html=True)

    else:
        st.write("Please input both input file path and image root path.")


if __name__ == "__main__":
    main()