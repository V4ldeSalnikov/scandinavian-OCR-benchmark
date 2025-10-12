from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator



class OCRDataset(ABC):
    id: str
    languages: list[str]
    default_evaluator : Evaluator | None


    @abstractmethod
    def load_dataset(self) -> Dataset:
        ...


