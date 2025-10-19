"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, GPT-OSS-20B, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod

from agents import MultiAgentModel, DummyModel
from utils.retry import retry_with_exponential_backoff

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass
    
    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """Run inference on the model
        
        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""
    
    def load(self, **kwargs):
        """Load ChatGPT model (Azure OpenAI). Environment tolerant."""

        # Normalize environment variants
        api_key = (
            os.getenv("AZURE_OPENAI_API_KEY_O1")
            or os.getenv("AZURE_OPENAI_KEY")
        )
        api_version = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
        azure_endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_ENDPOINT")
        )

        if not api_key or not azure_endpoint:
            raise ValueError("Azure OpenAI credentials missing. Set AZURE_OPENAI_API_KEY_O1 (or AZURE_OPENAI_KEY) and AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT).")
        
        from openai import AzureOpenAI
        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    @retry_with_exponential_backoff(
        max_retries=5,
        initial_sleep=2.0,
        max_sleep=60.0,
        exponential_base=2.0,
    )
    def _call_api_with_retry(self, request_kwargs: Dict[str, Any]) -> str:
        """Call OpenAI API with retry logic."""
        responses = self.model_client.chat.completions.create(**request_kwargs)
        return responses.choices[0].message.content

    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """ChatGPT inference"""
        kwargs = dict(kwargs or {})
        messages = [{"role": "user", "content": prompt}]
        model_lower = (self.model_name or "").lower()
        is_reasoning = self._is_reasoning_model_name(model_lower)

        # Extract generation parameters
        max_tokens_arg = kwargs.pop("max_tokens", max_tokens)
        max_completion_tokens = kwargs.pop("max_completion_tokens", max_tokens_arg)
        max_completion_tokens = max(1, min(int(max_completion_tokens), 8192))

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if is_reasoning:
            request_kwargs["max_completion_tokens"] = max_completion_tokens
            reasoning_effort = kwargs.pop("reasoning_effort", None)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            verbosity = kwargs.pop("verbosity", None)
            if verbosity:
                request_kwargs["verbosity"] = verbosity
        else:
            request_kwargs["max_tokens"] = int(max_tokens_arg)
            temperature = kwargs.pop("temperature", None)
            if temperature is not None:
                request_kwargs["temperature"] = float(temperature)
            top_p = kwargs.pop("top_p", None)
            if top_p is not None:
                request_kwargs["top_p"] = float(top_p)

        if "stop" in kwargs and kwargs["stop"] is not None:
            request_kwargs["stop"] = kwargs.pop("stop")
        if "seed" in kwargs and kwargs["seed"] is not None:
            request_kwargs["seed"] = kwargs.pop("seed")

        # Call the API with retry logic
        response = self._call_api_with_retry(request_kwargs)
        # print("\033[94m" + str(response) + "\033[0m")
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response}]
        
        return response, complete_messages

    @staticmethod
    def _is_reasoning_model_name(model_name: str) -> bool:
        if not model_name:
            return False
        reasoning_markers = ["gpt-5", "o1", "o3"]
        if any(marker in model_name for marker in reasoning_markers):
            if "gpt-5" in model_name and "chat" in model_name and "reasoning" not in model_name:
                return False
            return True
        return False


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""
    
    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                **kwargs,
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        print("messages:", messages)
        
        temperature = kwargs.get("temperature", 0.4)
        top_p = kwargs.get("top_p", 0.9)
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt', enable_thinking=False
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        print("response_text:", response_text)
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response_text}]
        
        return response_text, complete_messages


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""
    
    def __init__(self, model_name: str, model_instance, inference_func):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func
    
    def load(self, **kwargs):
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")
    
    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]
            
            response = self._inference_func(self.model, prompt, max_tokens)
            
            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]
            
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages

class GPTOSS20BModel(BaseModel):
    """GPT-OSS-20B wrapper"""

    def __init__(
        self,
        model_name: str,
        quantization: str = "auto",          # auto | fp16 | bf16 | 8bit
        reasoning_lvl: str = "medium",       # low | medium | high
        system_identity: str = None,         # optional system override
        developer_instructions: str = None,  # optional developer message
    ):
        super().__init__(model_name)
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.enc = None
        self.reasoning_lvl = reasoning_lvl
        self.system_identity = system_identity
        self.developer_instructions = developer_instructions

    def load(self, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization == "fp16":
            torch_dtype = torch.float16
            quant_config = None
        elif self.quantization == "bf16":
            torch_dtype = torch.bfloat16
            quant_config = None
        elif self.quantization == "8bit":
            torch_dtype = torch.bfloat16
            quant_config = None
        else:
            # this will automatically use MXFP4 weights.
            torch_dtype = "auto"
            quant_config = None

        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", **kwargs}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        temperature: float = kwargs.get("temperature", 1.0)
        top_p: float = kwargs.get("top_p", 1.0)
        builtin_tools: Optional[List[str]] = kwargs.get("builtin_tools")
        tools: Optional[List[dict]] = kwargs.get("tools")
        
        from openai_harmony import Role
        import logging
        from transformers import AutoTokenizer  

        # Build message list
        messages = []
        if self.system_identity or self.reasoning_lvl:
            sys_content = ""
            if self.system_identity:
                sys_content += self.system_identity + "\n"
            sys_content += f"Reasoning: {self.reasoning_lvl}."
            messages.append({"role": "system", "content": sys_content})

        if self.developer_instructions:
            messages.append({"role": "developer", "content": self.developer_instructions})

        messages.append({"role": "user", "content": prompt})

        # Apply Hugging Face chat template with fallback
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                    or "You are ChatGPT, a large language model trained by OpenAI.",
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)
        except Exception as e:
            logging.warning(
                f"[WARN] Custom chat_template in {self.model_name} failed "
                f"({type(e).__name__}: {e}). Falling back to base GPT-OSS template."
            )
            # Reload base tokenizer for Harmony
            base_tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            self.tokenizer.chat_template = base_tok.chat_template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                    or "You are ChatGPT, a large language model trained by OpenAI.",
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=(temperature>0),
            eos_token_id=None if not self.enc else self.enc.stop_tokens()[-1],
        )
        # Parse Harmony messages
        gen_tokens = outputs[0][input_ids.shape[-1]:].tolist()
 
        try:
            parsed = self.enc.parse_messages_from_completion_tokens(gen_tokens, role=Role.ASSISTANT)
            reasoning_trace = [msg.to_dict() for msg in parsed]

            # Prefer "final" channel
            finals = [msg for msg in parsed if msg.to_dict().get("channel") == "final"]
            if finals:
                final_response = "".join(c.text for c in finals[-1].content if hasattr(c, "text"))
            else:
                # Fallback: take last assistant message, but strip to short answer
                final_response = "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))

        except Exception as e:
            logging.error(f"[Harmony parse error] {e}")
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            final_response = text
            reasoning_trace = [{"role": "assistant", "content": text}]

        return final_response.strip(), reasoning_trace

class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit
        
        Args:
            output_dir: Directory to save results and submissions
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None
        
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        self.runtime = self.config.get('runtime', {}) if isinstance(self.config, dict) else {}
        self._telemetry_summary: Dict[str, Any] = {}
        self._inference_kwargs: Dict[str, Any] = {}
        
        self.output_dir = self.config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)
        self._is_reasoning = False
    
    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation
        
        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b-chat-hf")
            model_type: Type of model ("chatgpt", "local", "custom", "auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "gpt-oss-20b":
            self.model = GPTOSS20BModel(model_name)
        elif model_type == "gemini":
            try:
                from models.gemini_model import GeminiModel
            except Exception as e:
                logger.error(f"Failed to import GeminiModel: {e}")
                raise
            self.model = GeminiModel(model_name)
        elif model_type == "multiagent":
            # Multi-agent orchestrator defined below
            self.model = MultiAgentModel(model_name)
        elif model_type == "dummy":
            self.model = DummyModel(model_name or "dummy")
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "custom":
            # For custom models, user should provide model_instance and inference_func
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        # Pass config to allow multi-agent to configure roles
        try:
            self.model.load(config=self.config, **kwargs)
        except TypeError:
            self.model.load(**kwargs)
        
        self._is_reasoning = self.runtime.get('is_reasoning_model')
        if self._is_reasoning is None:
            self._is_reasoning = self._is_reasoning_model(self.model_name)
        self._inference_kwargs = self._prepare_inference_kwargs()
    
    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults
        
        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("Not config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("Not config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        name = (model_name or "").lower()
        if "gpt-oss-20b" in name:
            return "gpt-oss-20b"
        if "gemini" in name or "google-gemini" in name:
            return "gemini"
        if any(tok in name for tok in ["multiagent", "hybrid"]):
            return "multiagent"
        if "dummy" in name:
            return "dummy"
        if any(tok in name for tok in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        else:
            return "local"

    def _is_reasoning_model(self, model_name: Optional[str]) -> bool:
        if model_name is None:
            return False
        name = model_name.lower()
        reasoning_markers = ["gpt-5", "o1", "o3"]
        if any(marker in name for marker in reasoning_markers):
            # Exclude chat-only variants such as gpt-5-chat
            if "gpt-5" in name and "chat" in name and "reasoning" not in name:
                return False
            return True
        return False

    def _prepare_inference_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        runtime = self.runtime or {}
        max_tokens = int(runtime.get("max_tokens", 1024))
        kwargs["max_tokens"] = max_tokens

        if self._is_reasoning:
            kwargs["max_completion_tokens"] = int(runtime.get("max_completion_tokens", max_tokens))
            kwargs["reasoning_effort"] = runtime.get("reasoning_effort", "high")
            if runtime.get("verbosity"):
                kwargs["verbosity"] = runtime["verbosity"]
        else:
            if runtime.get("max_completion_tokens"):
                kwargs["max_completion_tokens"] = int(runtime.get("max_completion_tokens"))
            if runtime.get("temperature") is not None:
                kwargs["temperature"] = float(runtime["temperature"])
            if runtime.get("top_p") is not None:
                kwargs["top_p"] = float(runtime["top_p"])

        return kwargs

    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response"""
        if not response or response is None:
            return ""
            
        response = response.strip().upper()
        
        # Look for letter at the beginning
        if response and response[0] in ['A', 'B', 'C', 'D', 'E']:
            return response[0]
        
        # Look for "The answer is X" patterns
        import re
        patterns = [
            r"(?:answer is|answer:|is)\s*([ABCDE])",
            r"([ABCDE])\)",
            r"\b([ABCDE])\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # Default to empty string if nothing found (to avoid None values in CSV)
        return ""

    def _get_model_telemetry(self) -> Dict[str, Any]:
        if hasattr(self.model, "get_telemetry"):
            try:
                telemetry = self.model.get_telemetry()
                return telemetry or {}
            except Exception as exc:  # pragma: no cover - telemetry is optional
                logger.debug("Telemetry retrieval failed: %s", exc)
        return {}

    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        
        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile
        
        # Get metadata from various sources with priority order
        base_metadata = self.get_metadata(config_path, args, metadata)

        # Merge telemetry summary into metadata
        telemetry_summary = getattr(self, '_telemetry_summary', {})
        if telemetry_summary:
            # Update tool-related fields with telemetry data
            if 'average_tools_per_question' in telemetry_summary:
                base_metadata['average_tools_per_question'] = telemetry_summary['average_tools_per_question']
            if 'tool_category_coverage' in telemetry_summary:
                base_metadata['tool_category_coverage'] = telemetry_summary['tool_category_coverage']
            if 'average_tokens_per_question' in telemetry_summary:
                base_metadata['average_tokens_per_question'] = telemetry_summary['average_tokens_per_question']

        metadata = base_metadata
        
        # Create submission data for CSV
        submission_data = []

        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                
                # Clean up text fields to avoid CSV formatting issues
                prediction_text = prediction.get("open_ended_answer", "") or ""
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"
                else:
                    choice_clean = str(choice_raw).strip()
                
                # Ensure reasoning trace is not null
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE")
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        # Create DataFrame and save CSV with proper quoting and NaN handling
        if submission_data:
            df = pd.DataFrame(submission_data)
        else:
            logger.warning("No submission rows generated; creating empty submission with default columns")
            df = pd.DataFrame(columns=["id", "prediction", "choice", "reasoning"])
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',
            'reasoning': 'No reasoning available'
        }
        
        for col in df.columns:
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])
        
        csv_path = os.path.join(self.output_dir, filename)
        
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")
        
        logger.info("Performing final NULL check on choice column...")
        null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
        for pattern in null_patterns:
            count_before = (df['choice'] == pattern).sum()
            if count_before > 0:
                logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')
        
        empty_count = (df['choice'] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
            df['choice'] = df['choice'].replace('', 'NOTAVALUE')
        
        null_mask = df['choice'].isnull()
        if null_mask.sum() > 0:
            logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
            df.loc[null_mask, 'choice'] = 'NOTAVALUE'
        
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, filename)
            zipf.write(metadata_path, metadata_filename)
        
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")
        
        return zip_path

    def save_submission_with_metadata(self, results: List[EvaluationResult], 
                                     metadata: Dict = None, filename: str = "submission.csv",
                                     config_path: str = None, args: argparse.Namespace = None):
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package
        
        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used  
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        if metadata is None:
            telemetry = getattr(self, '_telemetry_summary', {}) or {}
            metadata = telemetry.copy()
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)

    def list_datasets(self):
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        _, ext = os.path.splitext(config_path)
        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
        metadata = config.get('metadata', config.get('meta_data', {}))
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")
        return metadata

    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        metadata = {}
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }
        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)
        return metadata

    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, 
                    fallback_metadata: Dict = None) -> Dict:
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }
        if fallback_metadata:
            metadata.update(fallback_metadata)
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")
        return metadata

    
    def evaluate(self, dataset_name: str, subset_size: int = None) -> EvaluationResult:
        """Evaluate model on a dataset, collecting telemetry when available."""
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")

        dataset = self._load_dataset(dataset_config)
        self._last_dataset_examples = dataset

        if subset_size is not None and subset_size > 0:
            dataset = dataset[:subset_size]
            logger.info(f"Subset size applied: {len(dataset)} examples")

        predictions: List[Dict] = []
        reasoning_traces: List[Any] = []
        failed_cases: List[Dict[str, Any]] = []  # Track failed cases
        total_count = len(dataset)
        accuracy_correct_count = 0
        accuracy_total_count = 0

        total_tool_calls = 0
        tool_names: set[str] = set()
        approx_token_counts: List[int] = []
        self._telemetry_summary = {}

        logger.info(f"Running evaluation on {total_count} examples...")
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            prediction = {"choice": "NOTAVALUE", "open_ended_answer": "Error"}
            reasoning_trace: Any = ""
            try:
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
            except Exception as exc:
                logger.error(f"Error processing example {idx}: {exc}")
                # Track failed case
                failed_cases.append({
                    "index": idx,
                    "id": example.get("id", f"example_{idx}"),
                    "question_type": example.get("question_type", "unknown"),
                    "error": str(exc),
                    "error_type": type(exc).__name__
                })

            predictions.append(prediction)
            reasoning_traces.append(reasoning_trace)

            telemetry = self._get_model_telemetry()
            if telemetry:
                calls = telemetry.get("tool_calls", []) or []
                total_tool_calls += len(calls)
                for call in calls:
                    # Extract tool name from different possible fields
                    tool_name = call.get("tool") or call.get("name") or call.get("provider") or "unknown"
                    tool_names.add(str(tool_name))

                    # Also track the provider to understand which provider was used
                    provider = call.get("provider", "unknown")
                    if provider != "unknown" and provider != "noop":
                        tool_names.add(f"{provider}:{tool_name}")

            approx_tokens = len(str(prediction.get("open_ended_answer", "")).split())
            approx_token_counts.append(approx_tokens)

            question_type = example["question_type"]
            expected_answer = example.get("answer", "")

            if question_type in {"multi_choice", "open_ended_multi_choice"}:
                accuracy_total_count += 1
                if expected_answer and prediction.get("choice") == expected_answer:
                    accuracy_correct_count += 1
            elif question_type == "open_ended":
                if expected_answer and prediction.get("open_ended_answer") == expected_answer:
                    accuracy_correct_count += 1

            if (idx + 1) % 10 == 0:
                current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                logger.info(f"Progress: {idx+1}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)")

        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0

        avg_tools = (total_tool_calls / total_count) if total_count else 0.0
        avg_tokens = (sum(approx_token_counts) / len(approx_token_counts)) if approx_token_counts else 0.0
        self._telemetry_summary = {
            "average_tools_per_question": f"{avg_tools:.2f}",
            "tool_category_coverage": ",".join(sorted(tool_names)),
            "average_tokens_per_question": f"{avg_tokens:.2f}",
        }

        logger.info(
            "Evaluation completed: %.2f%% accuracy (%d/%d) - excluding open-ended questions",
            accuracy * 100,
            accuracy_correct_count,
            accuracy_total_count,
        )
        logger.info("Total examples processed: %d", total_count)

        # Report failed cases if any
        if failed_cases:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"Failed Cases: {len(failed_cases)} out of {total_count} examples")
            logger.warning(f"{'='*60}")
            for failed_case in failed_cases:
                logger.warning(
                    f"Index {failed_case['index']} (ID: {failed_case['id']}, Type: {failed_case['question_type']}): "
                    f"{failed_case['error_type']} - {failed_case['error'][:100]}"
                )

            # Save failed cases to JSON file for reprocessing
            output_dir = dataset_config.get("output_dir", "competition_results")
            os.makedirs(output_dir, exist_ok=True)
            failed_cases_path = os.path.join(output_dir, "failed_cases.json")
            with open(failed_cases_path, "w") as f:
                json.dump(failed_cases, f, indent=2)
            logger.warning(f"Failed cases saved to: {failed_cases_path}")
            logger.warning(f"{'='*60}\n")

        # Store failed cases in the result object (add this to EvaluationResult later if needed)
        self._last_failed_cases = failed_cases

        return EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces,
        )
    
    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration with or without PyTorch."""
        from dataset_utils import build_dataset
        
        # Build dataset
        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )
        
        # Try to use DataLoader if torch is available
        try:
            from torch.utils.data import DataLoader  # type: ignore
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            iterator = (batch for batch in dataloader)
            use_torch = True
        except Exception:
            # Fallback: iterate indexably
            iterator = (dataset[i] for i in range(len(dataset)))
            use_torch = False
        
        dataset_list = []
        for item in iterator:
            if use_torch:
                question_type = item[0][0]
                if question_type == "multi_choice":
                    dataset_list.append({
                        "question_type": item[0][0],
                        "id": item[1][0],
                        "question": item[2][0],
                        "answer": item[3][0],
                    })
                elif question_type == "open_ended_multi_choice":
                    dataset_list.append({
                        "question_type": item[0][0],
                        "id": item[1][0],
                        "question": item[2][0],
                        "answer": item[3][0],
                        "meta_question": item[4][0],
                    })
                elif question_type == "open_ended":
                    dataset_list.append({
                        "question_type": item[0][0],
                        "id": item[1][0],
                        "question": item[2][0],
                        "answer": item[3][0],
                    })
            else:
                # item is a tuple as returned by CureBenchDataset.__getitem__
                question_type = item[0]
                if question_type == "multi_choice":
                    dataset_list.append({
                        "question_type": item[0],
                        "id": item[1],
                        "question": item[2],
                        "answer": item[3],
                    })
                elif question_type == "open_ended_multi_choice":
                    dataset_list.append({
                        "question_type": item[0],
                        "id": item[1],
                        "question": item[2],
                        "answer": item[3],
                        "meta_question": item[4],
                    })
                elif question_type == "open_ended":
                    dataset_list.append({
                        "question_type": item[0],
                        "id": item[1],
                        "question": item[2],
                        "answer": item[3],
                    })
        
        return dataset_list

    
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]
        
        # Format prompt
        if question_type == "multi_choice":
            prompt = f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:"
        elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
            prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:"
        
        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(prompt, **(self._inference_kwargs.copy()))
        
        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Then use meta question to get choice, if available
            if "meta_question" in example:
                # Prefer external mapper prompt if available
                mapper_inst = None
                try:
                    mapper_path = os.path.join(os.path.dirname(__file__), 'prompts', 'meta_choice_mapper.txt')
                    with open(mapper_path, 'r', encoding='utf-8') as f:
                        mapper_inst = f.read().strip()
                except Exception:
                    mapper_inst = None

                if mapper_inst:
                    meta_prompt = (
                        f"{mapper_inst}\n\n" 
                        f"{example['meta_question']}"
                        f"Agent's answer: {response.strip()}\n\nFinal choice:"
                    )
                else:
                    meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt, **(self._inference_kwargs.copy()))
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        return prediction, reasoning_trace
    
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        
        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile
        
        # Get metadata from various sources with priority order
        base_metadata = self.get_metadata(config_path, args, metadata)

        # Merge telemetry summary into metadata
        telemetry_summary = getattr(self, '_telemetry_summary', {})
        if telemetry_summary:
            # Update tool-related fields with telemetry data
            if 'average_tools_per_question' in telemetry_summary:
                base_metadata['average_tools_per_question'] = telemetry_summary['average_tools_per_question']
            if 'tool_category_coverage' in telemetry_summary:
                base_metadata['tool_category_coverage'] = telemetry_summary['tool_category_coverage']
            if 'average_tokens_per_question' in telemetry_summary:
                base_metadata['average_tokens_per_question'] = telemetry_summary['average_tokens_per_question']

        metadata = base_metadata
        
        # Create submission data for CSV
        submission_data = []
        
        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                # if result.reasoning_traces and i < len(result.reasoning_traces):
                #     trace = result.reasoning_traces[i]
                #     if isinstance(trace, list) and len(trace) > 0:
                #         # Convert list of messages to a simple text format
                #         text_parts = []
                #         for msg in trace:
                #             if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                #                 role = msg['role']
                #                 content = msg['content'].replace('\n', ' ').replace('\r', '').replace('"', "'")
                #                 text_parts.append(f"{role}: {content}")
                #         reasoning_trace = " | ".join(text_parts)
                #     else:
                #         # Fallback to string representation
                #         reasoning_trace = str(trace).replace('\n', ' ').replace('\r', '').replace('"', "'")
                
                # Clean up text fields to avoid CSV formatting issues
                prediction_text = prediction.get("open_ended_answer", "") or ""  # Ensure not None
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                
                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
                else:
                    choice_clean = str(choice_raw).strip()
                
                # Ensure reasoning trace is not null
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                # Create CSV row - let pandas handle the escaping
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                # Debug: Log if choice is NULL-like
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE")
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        # Create DataFrame and save CSV with proper quoting and NaN handling
        df = pd.DataFrame(submission_data)
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Aggressive null value cleaning
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',  # Use NOTAVALUE for choice instead of empty string
            'reasoning': 'No reasoning available'
        }
        
        # Replace all possible null-like values
        for col in df.columns:
            # Replace pandas null values
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            
            # Replace string representations of null
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            
            # Special handling for choice column - ensure it's never empty or null-like
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')  # Keep NOTAVALUE as is for choice
                # Replace any null-like values with NOTAVALUE
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                # Replace empty strings with NOTAVALUE for choice column
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')  # Also replace whitespace-only
            
            # Replace empty strings (except for choice column which can be empty)
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])  # Also replace whitespace-only
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Validate DataFrame before saving
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Final validation - check for any remaining nulls
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")
        
        # Check for any problematic data
        if not df.empty:
            for idx, row in df.head().iterrows():
                logger.debug(f"Sample row {idx}: id={row['id']}, choice='{row['choice']}', prediction_len={len(str(row['prediction']))}, reasoning_len={len(str(row['reasoning']))}")
        
        # Final safety check: ensure choice column has no NULL values or empty strings
        # Only process if DataFrame has a 'choice' column
        if 'choice' in df.columns:
            logger.info("Performing final NULL check on choice column...")
            null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
            for pattern in null_patterns:
                count_before = (df['choice'] == pattern).sum()
                if count_before > 0:
                    logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                    df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')

            # Replace empty strings with NOTAVALUE to avoid NULL validation issues
            empty_count = (df['choice'] == '').sum()
            if empty_count > 0:
                logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace('', 'NOTAVALUE')

            # Also replace any remaining pandas nulls in choice column
            null_mask = df['choice'].isnull()
            if null_mask.sum() > 0:
                logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
                df.loc[null_mask, 'choice'] = 'NOTAVALUE'
        else:
            logger.warning("DataFrame has no 'choice' column - likely no data was loaded")
        

        # Use proper CSV parameters for robust handling of complex data
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)  # index=False to avoid pandas index issues
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP file with CSV and metadata
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(csv_path, filename)
            # Add metadata JSON to zip
            zipf.write(metadata_path, metadata_filename)
        
        # Calculate and log overall accuracy
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")
        
        return zip_path
    
    def save_submission_with_metadata(self, results: List[EvaluationResult], 
                                     metadata: Dict = None, filename: str = "submission.csv",
                                     config_path: str = None, args: argparse.Namespace = None):
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package
        
        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used  
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # Use the stored dataset examples from the last evaluation
        dataset_examples = getattr(self, '_last_dataset_examples', [])

        if metadata is None:
            telemetry = getattr(self, '_telemetry_summary', {}) or {}
            metadata = {}
        else:
            telemetry = getattr(self, '_telemetry_summary', {}) or {}

        # Merge telemetry into metadata to ensure tool stats are included
        metadata.update(telemetry)

        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Metadata dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
        
        # Extract metadata from config
        metadata = config.get('metadata', config.get('meta_data', {}))
        
        # Validate required fields
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")
        
        return metadata
    
    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Map argument names to metadata fields
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }
        
        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)
        
        return metadata
    
    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, 
                    fallback_metadata: Dict = None) -> Dict:
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)
        
        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary
            
        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }
        
        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)
        
        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")
        
        return metadata

def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata
    
    Returns:
        ArgumentParser with metadata-related arguments
    """
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')
    
    # Model information
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'], 
                       help='Type of base model (API or OpenWeighted)')
    
    # Track information
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')
    
    # Dataset and submission info
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='competition_results', 
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv', 
                       help='Output CSV filename for submission (will be packaged in zip)')
    
    # Evaluation settings
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')
    
    return parser


def load_config_file(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args
    
    config = load_config_file(args.config)
    
    # First, handle the metadata section specially - merge its contents directly
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)
    
    add_config_to_args(config)
    return args
