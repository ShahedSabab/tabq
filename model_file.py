from transformers import pipeline
import pandas as pd

class TapasInference:
    def __init__(self, tqa, df):
        self.tqa = tqa
        self.flag = False
        self.table = df.astype(str)

    def load_model(self, model_name, df):
        try:
            self.tqa = pipeline(task="table-question-answering", model="google/"+model_name)
            self.table = df.astype(str)
            self.flag = True
        except Exception as es:
            print("Model loading error!!! ", es)

    def process_result(self, answer):
        a = answer['answer']
        result = None
        try:
            op = a.split('>')[0].strip()
            num_list = a.split('>')[1].split(',')
        except:
            result = a

        if op == "SUM":
            try:
                total = 0
                for i in range(len(num_list)):
                    total += float(num_list[i].strip())
                result = total
            except:
                pass

        elif op == "AVERAGE":
            try:
                total = 0
                for i in range(len(num_list)):
                    total += float(num_list[i].strip())
                avg = total / len(num_list)
                result = avg
            except:
                pass
        return result

    def infer(self, query):
        res = None
        answer = self.tqa(table=self.table, query=query)
        if type(answer) == list:
            for answer in answer:
                res = self.process_result(answer)
        else:
            res = self.process_result(answer)
        return res