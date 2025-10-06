
from pydantic_evals import Case, Dataset
from PIL import Image
from pathlib import Path
import models.model_interface
from models.model_interface import OCRInput


#creating simple dataset with two images
def create_dataset() -> Dataset :

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    IMG_DIR = ROOT / "test_images"

    img1_path = IMG_DIR / "test_image_1.jpg"
    img2_path = IMG_DIR / "test_image_2.jpg"
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    case1 = Case(
        name='first_case',
        inputs= OCRInput(image = img1) ,
        expected_output='Stå af og træk cyklen',
        metadata={'difficulty': 'easy'},
    )

    case2 = Case(
        name = "second_case",
        inputs = OCRInput(image = img2),
        expected_output = 'Knallert forbudt',
        metadata = {'difficulty': 'easy'},
    )

    dataset = Dataset(cases=[case1, case2])
    return dataset