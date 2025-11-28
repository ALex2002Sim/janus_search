from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModelService(ABC):
    """
    Базовый абстрактный класс для всех моделей.
    Обеспечивает единый интерфейс.
    """
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Преобразование входных данных в формат для модели."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """Преобразование выхода модели в человеко-читаемый формат."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Основной метод: от входа до предсказания."""
        pass