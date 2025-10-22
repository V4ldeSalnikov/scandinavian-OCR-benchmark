from models.models_implementations.easyocr_adapter import EasyOCRAdapter
from models.models_implementations.qwen_2b_vl_adapter import Qwen2bAdapter
from ocr_datasets.datasets.gothenburg_price_tag import GothenburgPriceTag
from ocr_datasets.datasets.historical_danish_handwriting import DanishHistoricalHandwriting
from ocr_datasets.datasets.norhand import Norhand
from ocr_datasets.datasets.simple_dataset import *
from evaluators.standard_evaluator import *
from tasks.ocr_task import *


def main():

    dataset_meta = DanishHistoricalHandwriting(max_examples = 1, streaming = True)
    dataset = dataset_meta.load_dataset()
    task = default_ocr_task(Qwen2bAdapter())
    report = dataset.evaluate_sync(task)
    report.print(include_input=True, include_output=True, include_durations=True)





if __name__ == "__main__":
    main()
