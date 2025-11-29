# base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseModelService(ABC):
    """
    Базовый абстрактный класс для всех ML-моделей.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Загрузка модели и токенайзера."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Преобразование входных данных."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """Преобразование результата."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Синхронный метод предсказания."""
        raise NotImplementedError
