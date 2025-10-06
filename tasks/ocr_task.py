from models.model_interface import OCRInput, OCROutput, OCRModel


def make_ocr_task(model):
    def task(inpt: OCRInput) -> str:
        return model(inpt).text
    return task