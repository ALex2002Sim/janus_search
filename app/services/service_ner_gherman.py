from typing import Any, Dict, List, Tuple
import re
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForTokenClassification
import asyncio
from fastapi.concurrency import run_in_threadpool
from functools import partial

from app.services.base import BaseModelService


class GhermanNerService(BaseModelService):
    def __init__(
        self, model_name: str = "Gherman/bert-base-NER-Russian", device: str = "cpu"
    ):
        super().__init__(model_name=model_name, device=device)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _merge_subword_tokens_and_labels(
        self, tokens: List[str], labels: List[str]
    ) -> List[Tuple[str, str]]:
        merged = []
        current_word = ""
        current_label = "O"

        for token, label in zip(tokens, labels):
            if token.startswith("[") and token.endswith("]"):
                continue

            if token.startswith("##"):
                current_word += token[2:]
                if label != "O":
                    current_label = label
            else:
                if current_word:
                    merged.append((current_word, current_label))
                current_word = token
                current_label = label

        if current_word:
            merged.append((current_word, current_label))

        return merged

    def preprocess(self, text: str) -> Dict[str, Any]:
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def postprocess(self, tokens: torch.Tensor, logits: torch.Tensor) -> str:
        probs = softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)[0].cpu()

        tokens_list = self.tokenizer.convert_ids_to_tokens(tokens[0].cpu())
        labels = [self.model.config.id2label[i.item()] for i in pred_ids]

        merged = self._merge_subword_tokens_and_labels(tokens_list, labels)

        address = {
            "COUNTRY": "",
            "REGION": "",
            "CITY": "",
            "DISTRICT": "",
            "STREET": "",
            "HOUSE": "",
        }

        for word, label in merged:
            if label != "O":
                entity = label.split("-")[-1]
                if entity in address:
                    address[entity] += f" {word}"

        result = " ".join(address.values()).strip()
        return re.sub(r"\s+", " ", result)

    def _predict_sync(self, text: str) -> str:
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.postprocess(inputs["input_ids"], outputs.logits)

    async def predict_async(self, input_data: str) -> str:
        return await run_in_threadpool(partial(self._predict_sync, input_data))

    def predict(self, input_data: str) -> str:
        return asyncio.run(self.predict_async(input_data))
