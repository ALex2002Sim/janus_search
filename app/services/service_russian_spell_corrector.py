from typing import Any, Dict
import asyncio
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.concurrency import run_in_threadpool
from app.services.base import BaseModelService


class RussianSpellCorrectorService(BaseModelService):
    """
    Spell corrector на основе T5 модели.
    Наследуется от BaseModelService.
    """

    def __init__(self, 
        model_name: str = "UrukHan/t5-russian-spell", 
        device: str | None = None,
        max_length: int = 256
        ):
        self.model_name = model_name
        self.max_length = max_length

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model()

    def _load_model(self) -> None:
        """
        Синхронная загрузка модели и токенайзера.
        (Можно вызвать отдельно для async загрузки в FastAPI)
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input_data: str) -> str:
        """
        Подготовка строки для модели.
        """
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string")
        return input_data.strip()

    async def _generate(self, text: str, gen_kwargs: dict) -> str:
        """
        Асинхронный вызов model.generate через threadpool.
        """
        
        def _work(tokenizer, model, input_text, gkwargs):
            batch = tokenizer(input_text, return_tensors="pt", padding=True).to(
                model.device
            )
            out = model.generate(**batch, **gkwargs)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        return await run_in_threadpool(
            partial(_work, self.tokenizer, self.model, text, gen_kwargs)
        )

    def postprocess(self, raw_output: str) -> str:
        """
        Простейшая пост-обработка: убираем лишние пробелы.
        """
        return raw_output.strip()

    async def predict_async(
        self,
        input_data: str,
        num_beams: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Dict[str, Any]:
        """
        Основной метод: от строки до исправленного результата.
        """
        preprocessed = self.preprocess(input_data)
        self._semaphore = asyncio.Semaphore()

        gen_kwargs = {
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        async with self._semaphore:
            raw_output = await self._generate(preprocessed, gen_kwargs)

        corrected = self.postprocess(raw_output)
        return corrected

    def predict(self, input_data: str) -> Dict[str, Any]:
        """
        Синхронный метод для совместимости с BaseModelService.
        Вызывает async predict_async через asyncio.run
        """
        return asyncio.run(self.predict_async(input_data))
