from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModelService(ABC):
    """
    Базовый абстрактный класс для всех моделей.
    Обеспечивает единый интерфейс.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Загрузка модели (реализуется в дочерних классах)."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Преобразование входных данных в формат для модели."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_output: Any) -> Any:
        """Преобразование выхода модели в человеко-читаемый формат."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Основной метод: от входа до предсказания."""
        raise NotImplementedError
