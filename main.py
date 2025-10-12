from models.models_implementations.qwen_2b_vl_adapter import Qwen2bAdapter
from ocr_datasets.test_dataset import *
from evaluators.test_evaluator import *
from models.models_implementations.easyocr_adapter import *
from tasks.ocr_task import *
import torch

def main():

    dataset = create_dataset()
    dataset.add_evaluator(TestEvaluator())
    task = make_ocr_task(Qwen2bAdapter())
    report = dataset.evaluate_sync(task)
    report.print(include_input=True, include_output=True, include_durations=True)



if __name__ == "__main__":
    main()
