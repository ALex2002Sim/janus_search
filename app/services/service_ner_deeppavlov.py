# from app.services.base import BaseModelService
# from deeppavlov import build_model
# from typing import Any, Dict, List, Tuple
# import torch

# class DeepPavlovNer(BaseModelService):
#     def __init__(self, model_name : str = "ner_collection3_bert"):
#         self.model_name = model_name
#         self.model = build_model(self.model_name, download=True, install=False)

#     def preprocess(self, text: str) -> str:
#         return text

#     def postprocess(self, model_res) -> str:
#         data = []
#         for i, j in zip(model_res[0][0], model_res[1][0]):
#             data.append((i, j))
#         return model_res

#     def predict(self, text) -> Dict[str, Any]:
#         res = self.model([text])
#         output = self.postprocess(res)
#         return output

# if __name__ == "__main__":
#     NerDeepPavl = DeepPavlovNer()
#     print(NerDeepPavl.predict("Ул. Калинина, район Москва купить автомобиль дом 5"))
