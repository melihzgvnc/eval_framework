from dataclasses import dataclass
from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.models import HuggingFaceModel

@dataclass
class NLIEvaluator(BaseEvaluator):
    def __init__(self, model_path: str):
        self.model = HuggingFaceModel(
            model_type="sequence_classification"
        )
        self.model.load(model_path)

    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        result = self.model.nli_predict(sample, answer)
        
        # Use the factory method for consistent NLI results
        return EvaluationResult.from_nli_result(result)