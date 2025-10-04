from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from PIL import Image
from dataclasses import dataclass

@dataclass
class OCRInput:
    image : Image.Image

@dataclass
class OCROutput:
    text : str

class OCRModel(ABC):
    id : str
    supported_tasks : list[str]

    @abstractmethod
    def __call__(self, inputs : OCRInput) -> OCROutput:
        ...
    def batch_call(self, inputs: list[OCRInput]) -> list[OCROutput]:
        return [self(inpt) for inpt in inputs]

