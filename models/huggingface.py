from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, PeftConfig
import numpy as np

from .base_model import Model


@dataclass
class HuggingFaceModel(Model):
    """HuggingFace model implementation with support for multiple model types.
    
    This class provides a concrete implementation for HuggingFace
    transformer models, including support for:
    - Causal Language Models (text generation)
    - Sequence Classification Models (NLI, sentiment, etc.)
    - PEFT (Parameter-Efficient Fine-Tuning) models
    
    The model type is automatically detected based on the model architecture.
    """
    
    model_type: str = field(default="auto")  # "auto", "causal", "sequence_classification"
    is_peft: bool = False
    base_model: Optional[str] = None
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: Optional[PreTrainedTokenizer] = field(default=None, init=False)
    model: Optional[PreTrainedModel] = field(default=None, init=False)
    generation_config: Optional[GenerationConfig] = field(default=None, init=False)
    num_labels: Optional[int] = field(default=None, init=False)
    
    def load(self, model_path: str) -> None:
        """Load a HuggingFace model from the specified path.
        
        Supports multiple model types:
        - Causal Language Models (AutoModelForCausalLM)
        - Sequence Classification Models (AutoModelForSequenceClassification)
        - PEFT models (with automatic base model detection)
        
        Args:
            model_path: Path to the model (can be HuggingFace hub ID or local path)
            
        Raises:
            ValueError: If model loading fails
            ImportError: If required packages are not installed
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine model type if auto
            model_type = self._detect_model_type(model_path)
            
            # Load model based on type
            if self.is_peft:
                # Load PEFT model
                peft_config = PeftConfig.from_pretrained(model_path)
                base_model_path = peft_config.base_model_name_or_path
                
                if model_type == "causal":
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                elif model_type == "sequence_classification":
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                else:
                    raise ValueError(f"Unsupported model type for PEFT: {model_type}")
                
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.base_model = base_model_path
                
            else:
                # Load base model
                if model_type == "causal":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    # Try to load generation config for causal models
                    try:
                        self.generation_config = GenerationConfig.from_pretrained(model_path)
                    except:
                        self.generation_config = None
                        
                elif model_type == "sequence_classification":
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    # Get number of labels for classification models
                    self.num_labels = self.model.config.num_labels
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            # Move to device if not using device_map
            if self.device != "cuda" or self.model.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Update model state
            self.model_path = model_path
            self.is_loaded = True
            self.model_type = model_type
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect the type of model based on its configuration.
        
        Args:
            model_path: Path to the model
            
        Returns:
            Model type: "causal" or "sequence_classification"
            
        Raises:
            ValueError: If model type cannot be determined
        """
        if self.model_type != "auto":
            return self.model_type
        
        try:
            # Try to load config to determine model type
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            
            # Check if it's a causal LM
            if hasattr(config, "is_decoder") and config.is_decoder:
                return "causal"
            
            # Check if it's a sequence classification model
            if hasattr(config, "architectures"):
                archs = config.architectures
                if any("ForSequenceClassification" in arch for arch in archs):
                    return "sequence_classification"
                elif any("ForCausalLM" in arch for arch in archs):
                    return "causal"
            
            # Default based on common patterns
            if hasattr(config, "vocab_size") and hasattr(config, "n_positions"):
                return "causal"
            else:
                return "sequence_classification"
                
        except Exception:
            # If we can't determine, default to causal for backward compatibility
            return "causal"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the given prompt (for causal models).
        
        Args:
            prompt: Input text to generate from
            **kwargs: Additional generation parameters that override default config
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model is not loaded or not a causal model
            ValueError: If generation fails
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be loaded before generating")
        
        if self.model_type != "causal":
            raise RuntimeError(f"generate() is only available for causal models, not {self.model_type}")
        
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation config
            gen_config = self.generation_config.to_dict() if self.generation_config else {}
            gen_config.update(kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the input prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")
    
    def classify(self, texts: Union[str, List[str]], **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify text(s) using a sequence classification model (e.g., NLI).
        
        Args:
            texts: Single text string or list of texts to classify
            **kwargs: Additional classification parameters
            
        Returns:
            For single text: Dictionary with classification results
            For multiple texts: List of dictionaries with classification results
            
        Raises:
            RuntimeError: If model is not loaded or not a sequence classification model
            ValueError: If classification fails
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be loaded before classifying")
        
        if self.model_type != "sequence_classification":
            raise RuntimeError(f"classify() is only available for sequence classification models, not {self.model_type}")
        
        try:
            # Handle single text or list of texts
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=kwargs.get("max_length", 512)
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Classify
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            # Convert to CPU numpy for easier handling
            probabilities = probabilities.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Prepare results
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                result = {
                    "prediction": int(pred),
                    "probabilities": probs.tolist(),
                    "confidence": float(np.max(probs)),
                    "text": texts[i]
                }
                
                # Add label mapping if available
                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    label = self.model.config.id2label.get(int(pred), f"label_{int(pred)}")
                    result["label"] = label
                
                results.append(result)
            
            return results[0] if single_input else results
            
        except Exception as e:
            raise ValueError(f"Classification failed: {str(e)}")
    
    def nli_predict(self, premise: str, hypothesis: str, **kwargs) -> Dict[str, Any]:
        """Perform Natural Language Inference prediction.
        
        This is a specialized method for NLI models that takes a premise
        and hypothesis and returns NLI predictions (entailment, contradiction, neutral).
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with NLI prediction results
            
        Raises:
            RuntimeError: If model is not loaded or not a sequence classification model
            ValueError: If NLI prediction fails
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be loaded before NLI prediction")
        
        if self.model_type != "sequence_classification":
            raise RuntimeError(f"nli_predict() is only available for sequence classification models, not {self.model_type}")
        
        try:
            # Format for NLI (some models use special formatting)
            if hasattr(self.tokenizer, "build_inputs_with_special_tokens"):
                # Use model-specific formatting if available
                input_text = f"{premise} {self.tokenizer.sep_token} {hypothesis}"
            else:
                # Default formatting
                input_text = f"{premise} [SEP] {hypothesis}"
            
            # Classify
            result = self.classify(input_text, **kwargs)
            
            # Add NLI-specific information
            if isinstance(result, dict):
                result["premise"] = premise
                result["hypothesis"] = hypothesis
                
                # Map NLI labels if available
                if "label" in result:
                    label = result["label"].lower()
                    if "entail" in label:
                        result["nli_relation"] = "entailment"
                    elif "contradict" in label:
                        result["nli_relation"] = "contradiction"
                    elif "neutral" in label:
                        result["nli_relation"] = "neutral"
                    else:
                        result["nli_relation"] = "unknown"
            
            return result
            
        except Exception as e:
            raise ValueError(f"NLI prediction failed: {str(e)}")
