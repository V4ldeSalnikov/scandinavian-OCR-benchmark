
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from metrics.metrics import wer, cer

#creating simple evaluator to test if it works
@dataclass
class StandardEvaluator(Evaluator):
    """
    Standard Evaluator class that measures Character and Word error rates between OCR model output and ground truth
    """
    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, float]:
        return {
            "cer": cer(ctx.output, ctx.expected_output),
            "wer": wer(ctx.output, ctx.expected_output),
        }
