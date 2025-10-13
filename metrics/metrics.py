
from evaluate import load

# CER - Character error rate metrics
def cer(model_output : str, ground_truth : str) -> int :
    metrics = load("cer")
    cer_score = metrics.compute(predictions=[model_output], references=[ground_truth])
    return cer_score

# WER - word error rate metrics
def wer (model_output : str, ground_truth : str) -> int :
    metrics = load("wer")
    wer_score = metrics.compute(predictions=[model_output], references=[ground_truth])
    return wer_score

