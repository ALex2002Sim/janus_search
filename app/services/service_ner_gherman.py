from app.services.base import BaseModelService
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from typing import Any, Dict, List, Tuple
from torch.nn.functional import softmax


class GhermanNerService(BaseModelService):
    def __init__(self, model_name: str = "Gherman/bert-base-NER-Russian"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

    def _merge_subword_tokens_and_labels(
        self, tokens: List[str], labels: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Объединяет субтокены (##) и их метки.
        Очищает от мусорных токенов, которые нужны для разметки модели [CLS] [SEP] [PAD]
        """
        merged = []
        current_word = ""
        current_label = "O"

        for token, label in zip(tokens, labels):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
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

    def preprocess(self, text: str) -> dict:
        """Преобразует текст в тензоры для модели."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )

    def postprocess(self, tokens: torch.Tensor, logits: torch.Tensor) -> str:
        """Преобразует выход модели в строку адреса."""
        probs = softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)[0].cpu()
        tokens_list = self.tokenizer.convert_ids_to_tokens(tokens[0].cpu())
        predicted_labels = [self.model.config.id2label[idx.item()] for idx in pred_ids]

        merged = self._merge_subword_tokens_and_labels(tokens_list, predicted_labels)

        address_parts = {
            "COUNTRY": "",
            "REGION": "",
            "CITY": "",
            "DISTRICT": "",
            "STREET": "",
            "HOUSE": "",
        }

        for word, label in merged:
            if label != "O":
                entity_type = label.split("-")[-1]
                if entity_type in address_parts:
                    address_parts[entity_type] += " " + word

        result = " ".join(address_parts.values()).strip()
        result = re.sub(r"\s+", " ", result)
        return result

    def predict(self, text: str) -> Dict[str, Any]:
        """Основной метод: от текста до строки адреса."""
        inputs = self.preprocess(text)

        with torch.no_grad():
            outputs = self.model(**inputs)

        address_str = self.postprocess(inputs["input_ids"], outputs.logits)

        return address_str


if __name__ == "__main__":
    GhMod = GhermanNerService()
    print(GhMod.predict("Ул. Калинина, район Москва купить автомобиль дом 5"))
