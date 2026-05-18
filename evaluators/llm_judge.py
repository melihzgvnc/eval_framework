from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.models import OpenAIModel


@dataclass
class LLMJudge(BaseEvaluator):
    """LLM-as-a-Judge evaluator with structured JSON output.
    
    Uses an LLM to evaluate answers based on various criteria
    (factuality, helpfulness, coherence, etc.) with reliable
    structured JSON responses.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        evaluation_criteria: str = "factuality",
        threshold: float = 0.7,  # Score threshold for passing
        use_structured_output: bool = True,  # Use JSON mode for reliable parsing
    ):
        self.model = OpenAIModel(
            model=model_name,
            api_key=api_key,
            use_responses_api=False,  # Use Chat Completions for structured output
        )
        self.evaluation_criteria = evaluation_criteria
        self.threshold = threshold
        self.use_structured_output = use_structured_output
        
        # Define JSON schema for structured output
        self.judgment_schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "Overall score from 0-100",
                    "minimum": 0,
                    "maximum": 100
                },
                "reasoning": {
                    "type": "string", 
                    "description": "Detailed reasoning for the score"
                },
                "sub_scores": {
                    "type": "object",
                    "description": "Optional breakdown scores by dimension",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the judgment (0-1)",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["score", "reasoning"]
        }
        
        # Define scoring rubrics for different criteria
        self.rubrics = {
            "factuality": """
            Evaluate the factuality of this answer based on the provided context.
            
            CONTEXT:
            {context}
            
            ANSWER:
            {answer}
            
            SCORING GUIDELINES:
            - 0-20: Completely false/made up, contradicts context
            - 21-40: Mostly false with minor truthful elements  
            - 41-60: Partially true with significant errors/omissions
            - 61-80: Mostly true with minor inaccuracies
            - 81-100: Completely accurate and well-supported by context
            
            Return a JSON object with your evaluation.
            """,
            
            "helpfulness": """
            Evaluate how helpful this answer is for the user's question.
            
            QUESTION:
            {question}
            
            ANSWER:
            {answer}
            
            SCORING GUIDELINES:
            - 0-20: Not helpful at all, irrelevant or misleading
            - 21-40: Slightly helpful but incomplete or vague
            - 41-60: Somewhat helpful but has significant issues
            - 61-80: Helpful and mostly addresses the question
            - 81-100: Extremely helpful, complete, and insightful
            
            Return a JSON object with your evaluation.
            """,
            
            "coherence": """
            Evaluate the coherence and clarity of this answer.
            
            ANSWER:
            {answer}
            
            SCORING GUIDELINES:
            - 0-20: Incoherent, confusing, unreadable
            - 21-40: Poorly organized, hard to follow
            - 41-60: Somewhat clear but with organizational issues
            - 61-80: Clear and well-organized
            - 81-100: Exceptionally clear, logical, and easy to follow
            
            Return a JSON object with your evaluation.
            """,
            
            "comprehensiveness": """
            Evaluate how comprehensive and thorough this answer is.
            
            QUESTION:
            {question}
            
            ANSWER:
            {answer}
            
            SCORING GUIDELINES:
            - 0-20: Very superficial, misses key points
            - 21-40: Covers some aspects but misses important details
            - 41-60: Moderately comprehensive with gaps
            - 61-80: Comprehensive covering most important aspects
            - 81-100: Exceptionally thorough and complete
            
            Return a JSON object with your evaluation.
            """,
        }
    
    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        """Evaluate an answer using LLM-as-a-Judge with structured output.
        
        Args:
            sample: Context or question
            answer: Model-generated answer to evaluate
            
        Returns:
            EvaluationResult with LLM Judge details
            
        Raises:
            ValueError: If structured output parsing fails
        """
        # Get the appropriate rubric
        rubric = self.rubrics.get(
            self.evaluation_criteria, 
            self.rubrics["factuality"]  # Default to factuality
        )
        
        # Format the prompt
        prompt = rubric.format(context=sample, question=sample, answer=answer)
        
        # System message for structured output
        system_message = """You are an expert evaluator. 
        Always return your evaluation as a valid JSON object.
        Be objective, thorough, and consistent in your scoring."""
        
        try:
            if self.use_structured_output:
                # Use structured JSON output
                judgment = self.model.generate_structured(
                    prompt=prompt,
                    schema=self.judgment_schema,
                    system=system_message,
                    temperature=0.1,
                    max_tokens=500,
                )
                
                # Extract fields from structured response
                score = judgment.get("score", 50.0)
                reasoning = judgment.get("reasoning", "No reasoning provided")
                sub_scores = judgment.get("sub_scores")
                confidence = judgment.get("confidence", 0.8)
                raw_response = json.dumps(judgment, indent=2)
                
            else:
                # Fallback to text parsing (for backward compatibility)
                response = self.model.generate(
                    prompt=prompt,
                    system=system_message,
                    temperature=0.1,
                    max_tokens=500,
                )
                raw_response = response
                
                # Parse text response
                score, reasoning, sub_scores, confidence = self._parse_text_judgment(response)
        
        except Exception as e:
            raise ValueError(f"LLM Judge evaluation failed: {str(e)}")
        
        # Normalize score to 0-1 range for consistency
        normalized_score = score / 100.0
        
        # Create evaluation result
        return EvaluationResult.from_llm_judge_result(
            score=normalized_score,
            reasoning=reasoning,
            sub_scores=sub_scores,
            raw_response=raw_response,
            threshold=self.threshold,
            evaluation_criteria=self.evaluation_criteria,
            sample=sample,
            answer=answer,
            confidence=confidence,
        )
    
    def _parse_text_judgment(self, response: str) -> tuple[float, str, Optional[Dict[str, float]], float]:
        """Parse text-based LLM judgment (fallback method).
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (score, reasoning, sub_scores, confidence)
        """
        # Default values
        score = 50.0  # Default middle score
        reasoning = response
        sub_scores = None
        confidence = 0.5  # Default confidence
        
        # Try to extract structured data even from text
        import re
        
        # First, try to find JSON in the response
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        if json_matches:
            # Try to parse the largest JSON block
            for json_str in sorted(json_matches, key=len, reverse=True):
                try:
                    data = json.loads(json_str)
                    if "score" in data:
                        score = float(data.get("score", score))
                        reasoning = data.get("reasoning", reasoning)
                        sub_scores = data.get("sub_scores", sub_scores)
                        confidence = data.get("confidence", confidence)
                        break
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try pattern matching
        if score == 50.0:  # Still default
            # Look for "SCORE: X" pattern
            score_pattern = r"SCORE:\s*(\d+(?:\.\d+)?)"
            match = re.search(score_pattern, response, re.IGNORECASE)
            
            if match:
                try:
                    score = float(match.group(1))
                except ValueError:
                    pass
        
        # Look for multiple scores
        multi_score_pattern = r"(\w+):\s*(\d+(?:\.\d+)?)"
        all_matches = re.findall(multi_score_pattern, response)
        
        if len(all_matches) > 1:
            sub_scores = {}
            for name, value in all_matches:
                try:
                    sub_scores[name.lower()] = float(value) / 100.0  # Normalize
                except ValueError:
                    pass
        
        return score, reasoning, sub_scores, confidence
    
    def evaluate_multi_criteria(
        self, 
        sample: str, 
        answer: str,
        criteria_list: Optional[list[str]] = None
    ) -> Dict[str, EvaluationResult]:
        """Evaluate using multiple criteria.
        
        Args:
            sample: Context or question
            answer: Model-generated answer
            criteria_list: List of criteria to evaluate
            
        Returns:
            Dictionary mapping criteria to EvaluationResult
        """
        if criteria_list is None:
            criteria_list = ["factuality", "helpfulness", "coherence"]
        
        results = {}
        original_criteria = self.evaluation_criteria
        
        for criteria in criteria_list:
            self.evaluation_criteria = criteria
            results[criteria] = self.evaluate(sample, answer)
        
        # Restore original criteria
        self.evaluation_criteria = original_criteria
        
        return results