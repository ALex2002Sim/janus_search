# service_russian_spell_corrector.py
from typing import Any, Dict
import asyncio
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.concurrency import run_in_threadpool

from app.services.base import BaseModelService


class RussianSpellCorrectorService(BaseModelService):
    def __init__(
        self,
        model_name: str = "UrukHan/t5-russian-spell",
        device: str | None = None,
        max_length: int = 256,
    ):
        self.max_length = max_length
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(model_name=model_name, device=device)

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input_data: str) -> str:
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string")
        return input_data.strip()

    async def _generate_async(self, text: str, gen_kwargs: dict) -> str:
        def _generate(tokenizer, model, inp, kwargs):
            batch = tokenizer(inp, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**batch, **kwargs)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        return await run_in_threadpool(
            partial(_generate, self.tokenizer, self.model, text, gen_kwargs)
        )

    def postprocess(self, raw_output: str) -> str:
        return raw_output.strip()

    async def predict_async(self, text: str) -> str:
        pre = self.preprocess(text)

        gen_kwargs = {"max_length": self.max_length, "num_beams": 4}

        raw = await self._generate_async(pre, gen_kwargs)
        return self.postprocess(raw)

    def predict(self, input_data: str) -> str:
        return asyncio.run(self.predict_async(input_data))
