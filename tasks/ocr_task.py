from models.model_interface import OCRInput, OCROutput, OCRModel


def default_ocr_task(model:OCRModel):
    def task(inpt: OCRInput) -> str:
        return model(inpt).text
    return task