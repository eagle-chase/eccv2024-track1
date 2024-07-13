from eval.evaluation.stage1_eval_batch import GPTBatcher as GPTBatcher1
from eval.evaluation.stage2_eval_batch import GPTBatcher as GPTBatcher2
from eval.evaluation.stage3_eval_batch import GPTBatcher as GPTBatcher3



OPENAI_KEY='sk-xWQahTx28GvonjjC01F2C292581a4073A812D418Df90Dd2d'
API_BASE_URL='https://api.gpts.vin/v1'
EVAL_MODEL_NAME = "gpt-4o-2024-05-13"

class EvalModel1():
    def __init__(self) -> None:
        self.batcher = GPTBatcher2(
            api_key=OPENAI_KEY, 
            model_name=EVAL_MODEL_NAME, 
            num_workers=32,
            api_base_url=API_BASE_URL
        )
    def __call__(self, infer_result: str, answer: str) -> dict:
        print(f"{'='*30} Using eval model 1 {'='*30}")
        
        message = {
            "prediction": infer_result,
            "reference": answer,
        }
        ret = self.batcher.create_messages(message)
        
        results = self.batcher.handle_message_list([ret])
        output = results[0]
       
        try:
            score = int(output.split("Rating: [[")[1].split("]]")[0])
        except:
            try:
                score = int(output.split("rating is: [[")[1].split("]]")[0])
            except:
                try:
                    score = int(output.split("[[")[1].split("]]")[0])
                except:
                    print(f"Missing extract score")
                    score = -1
        return output, score

class EvalModel2():
    def __init__(self) -> None:
        self.batcher = GPTBatcher2(
            api_key=OPENAI_KEY, 
            model_name=EVAL_MODEL_NAME, 
            num_workers=32,
            api_base_url=API_BASE_URL
        )
    def __call__(self, infer_result: str, answer: str):
        print(f"{'='*30} Using eval model 2 {'='*30}")
        
        message = {
            "prediction": infer_result,
            "reference": answer,
        }
        ret = self.batcher.create_messages(message)
        
        results = self.batcher.handle_message_list([ret])
        output = results[0]
          
        try:
            score = int(output.split("Rating: [[")[1].split("]]")[0])
        except:
            try:
                score = int(output.split("rating is: [[")[1].split("]]")[0])
            except:
                try:
                    score = int(output.split("[[")[1].split("]]")[0])
                except:
                    print(f"Missing extract score")
                    score = -1
        return output, score

class EvalModel3():
    def __init__(self) -> None:
        self.batcher = GPTBatcher3(
            api_key=OPENAI_KEY, 
            model_name=EVAL_MODEL_NAME, 
            num_workers=32,
            api_base_url=API_BASE_URL,
        )
    def __call__(self, infer_result: str, answer: str) -> dict:
        print(f"{'='*30} Using eval model 3 {'='*30}")
        message = {
            "prediction": infer_result,
            "reference": answer,
        }
        ret = self.batcher.create_messages(message)
        results = self.batcher.handle_message_list([ret])
        output = results[0]

        try:
            score = int(output.split("Rating: [[")[1].split("]]")[0])
        except:
            try:
                score = int(output.split("rating is: [[")[1].split("]]")[0])
            except:
                try:
                    score = int(output.split("[[")[1].split("]]")[0])
                except:
                    print(f"Missing extract score")
                    score = -1
        return output, score
