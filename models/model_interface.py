from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from PIL import Image

class OCRInput:
    images : list[Image.Image]
class OCROutput:
    text : list[str]

class OCRModel(BaseModel):
    id : str
    supported_tasks : list[str]

    @abstractmethod
    def __call__(self, inputs : OCRInput) -> OCROutput:
        ...


