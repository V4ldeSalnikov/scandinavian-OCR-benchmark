import cv2
import easyocr
import numpy as np
from models.model_interface import OCRModel, OCRInput, OCROutput


class EasyOCRAdapter(OCRModel):

    def __init__(self):
        self.id = "easyocr"
        self._reader = easyocr.Reader(['da'])


    def __call__(self, inputs : OCRInput) -> OCROutput:

        #Convert image to OpenCV format

        img = inputs.image.convert("RGB")
        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)



        #detail = 0 will output only text, paragraph = True
        result = self._reader.readtext(cv_image, detail = 0, paragraph = True )

        text = ""
        for res in result :
            text += res + " "

        #delete last space from string
        text = text [:-1]

        return OCROutput(text)
