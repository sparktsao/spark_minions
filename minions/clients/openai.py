import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import openai

from minions.usage import Usage
from logging.handlers import RotatingFileHandler

LOG_FILE = "application.log"

# TODO: define one dataclass for what is returned from all the clients
class OpenAIClient:
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 4096):
        self.model_name = model_name
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.base_url = "https://api.rdsec.trendmicro.com/prod/aiendpoint/v1/"
        self.logger = logging.getLogger("OpenAIClient")
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=10*1024*1024, backupCount=1, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)

            # Define log format
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        self.temperature = temperature
        self.max_tokens = max_tokens



    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions using the OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to openai.chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                #"max_completion_tokens": 16000,
                **kwargs,
            }            

            # Only add temperature if NOT using the reasoning models (e.g., o3-mini model)
            if "o1" not in self.model_name and "o3" not in self.model_name:
                params["temperature"] = self.temperature

            response = openai.chat.completions.create(**params)
            
            self.logger.debug(str(params))
            self.logger.debug(response)
            

        except Exception as e:
            self.logger.error(f"Error during OpenAI API call: {e}")
            self.logger.debug(str(params))
            raise


        # Extract usage information
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        # The content is now nested under message
        return [choice.message.content for choice in response.choices], usage
