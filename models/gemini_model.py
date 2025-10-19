import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retry import retry_with_exponential_backoff


class GeminiModel:
    """Gemini wrapper using Vertex AI SDK with service account authentication.

    This implementation uses google-cloud-aiplatform for production-ready
    authentication with service accounts instead of API keys.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        # Use mapping or default to the provided name
        self.model_name = "gemini-2.5-pro"
        self._model = None
        self.project_id = None
        self.location = "us-central1"  # Default location for Vertex AI

    def load(self, **kwargs):
        """Load and initialize the Vertex AI Gemini model with service account auth."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from google.oauth2 import service_account
        except ImportError as e:
            raise ImportError(
                f"google-cloud-aiplatform not available: {e}. "
                "Install with: pip install google-cloud-aiplatform"
            )

        # Get project ID from environment
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "GOOGLE_PROJECT_ID not set. Export GOOGLE_PROJECT_ID to use Vertex AI."
            )

        # Get service account credentials path
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS not set. "
                "Export GOOGLE_APPLICATION_CREDENTIALS pointing to your service account JSON file."
            )

        # Check if credentials file exists
        if not os.path.exists(cred_path):
            raise FileNotFoundError(
                f"Service account credentials file not found at: {cred_path}"
            )

        try:
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                cred_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Initialize Vertex AI with credentials
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )

            # Create the generative model
            self._model = GenerativeModel(self.model_name)

            print(f"Successfully initialized Vertex AI Gemini model: {self.model_name}")
            print(f"Project: {self.project_id}, Location: {self.location}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Vertex AI: {e}\n"
                f"Ensure your service account has the 'Vertex AI User' role."
            )

    @retry_with_exponential_backoff(
        max_retries=5,
        initial_sleep=2.0,
        max_sleep=60.0,
        exponential_base=2.0,
        retryable_error_codes=(429, 503, 500, 502, 504),
    )
    def _generate_with_retry(self, prompt: str, generation_config: Dict) -> str:
        """Internal method to generate content with retry logic.

        Args:
            prompt: The input prompt
            generation_config: Generation configuration dictionary

        Returns:
            Generated text

        Raises:
            Exception: If generation fails after all retries
        """
        response = self._model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=False
        )

        # Extract text from response
        if response and response.text:
            return response.text
        else:
            return ""

    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """Generate text using the Vertex AI Gemini model.

        Args:
            prompt: The input prompt for generation
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters (temperature, top_p, top_k, etc.)

        Returns:
            Tuple of (generated_text, message_history)
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Configure generation parameters - use kwargs if provided, otherwise use defaults
            # Extract generation parameters from kwargs
            temperature = kwargs.get('temperature', 0.3)
            top_p = kwargs.get('top_p', 0.9)
            top_k = kwargs.get('top_k', 40)

            # Handle max_tokens vs max_completion_tokens (for consistency with other models)
            max_output = kwargs.get('max_completion_tokens', kwargs.get('max_tokens', max_tokens))

            generation_config = {
                "max_output_tokens": max_output,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }

            # Generate content with retry logic
            text = self._generate_with_retry(prompt, generation_config)

        except Exception as e:
            # All retries failed
            text = f"Error during inference after retries: {str(e)}"
            print(f"Gemini inference error after all retries: {e}")
            # Re-raise the exception so it can be caught and tracked by eval_framework
            raise

        # Format messages for compatibility with the framework
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]

        return text, messages