import json
import os
import time
from typing import List, Tuple

import requests

from etalon.core.llm_clients.base_llm_client import BaseLLMClient
from etalon.core.request_config import RequestConfig
from etalon.logger import init_logger
from etalon.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)

# Maximum number of responses to store for token counting
MAX_RESPONSES_ALLOWED_TO_STORE = 5


class OllamaChatCompletionsClient(BaseLLMClient):
    """Client for Ollama Chat Completions API."""

    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        super().__init__(model_name, tokenizer_name)
        self.address = os.environ.get("OLLAMA_API_BASE")
        if not self.address:
            self.address = "http://localhost:11434"
            logger.warning(
                "Warning: OLLAMA_API_BASE environment variable not set. Defaulting to localhost:11434."
            )
        self.start_time = time.monotonic()

    def total_tokens(self, response_list: List[str]) -> int:
        merged_content = "".join(response_list)
        return self.get_token_length(merged_content)

    def get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> Tuple[int, int]:
        previous_responses.append(current_response)
        current_tokens_received = (
            self.total_tokens(previous_responses) - previous_token_count
        )
        if len(previous_responses) > MAX_RESPONSES_ALLOWED_TO_STORE:
            previous_responses.pop(0)
        previous_token_count = self.total_tokens(previous_responses)
        return current_tokens_received, previous_token_count

    def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, str]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        # Format message for Ollama API
        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }

        # Add sampling parameters if provided
        sampling_params = request_config.sampling_params
        if sampling_params:
            # Map OpenAI params to Ollama equivalents
            ollama_params = {}

            # Direct mappings (parameters with same names)
            for param in ["temperature", "top_p", "top_k"]:
                if param in sampling_params:
                    ollama_params[param] = sampling_params[param]

            # Map max_tokens to num_predict if present
            if "max_tokens" in sampling_params:
                ollama_params["num_predict"] = sampling_params["max_tokens"]

            body.update(ollama_params)

        address = self.address
        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += "api/chat"  # Ollama chat endpoint

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()
        request_dispatched_at = time.monotonic() - self.start_time

        print(f"Sending request to ```{address}``` with body: ```{body}```")
        
        try:
            with requests.post(
                address, json=body, timeout=None, stream=True
            ) as response:
                if response.status_code != 200:
                    error_response_code = response.status_code
                    error_msg = response.text
                    logger.error(f"Request Error: {error_msg}")
                    response.raise_for_status()

                for line in response.iter_lines(chunk_size=None):
                    if not line:
                        continue
                    

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.error(f"JSON decode error with line: {line}")
                        continue  # Skip malformed JSON

                    if "error" in data:
                        error_msg = data["error"]
                        error_response_code = 500  # Default error code for Ollama
                        raise RuntimeError(f"Ollama API error: {error_msg}")

                    # Handle Ollama response format
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        # For streaming responses, the content will be partial
                        (
                            current_tokens_received,
                            previous_token_count,
                        ) = self.get_current_tokens_received(
                            previous_responses=previous_responses,
                            current_response=content,
                            previous_token_count=previous_token_count,
                        )

                        tokens_received += current_tokens_received
                        inter_token_times.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                        if current_tokens_received > 1:
                            inter_token_times.extend(
                                [0] * (current_tokens_received - 1)
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += content

                    # Check if response is done
                    if "done" in data and data["done"]:
                        break

        except Exception as e:
            logger.error(f"Warning Or Error: ({error_response_code}) {e}")
        
        print(f"Request completed with {tokens_received} tokens received.")
        print(f"Debug metrics:")
        print(f"  request_dispatched_at: {request_dispatched_at}")
        print(f"  inter_token_times: {inter_token_times[:5]}... (total: {len(inter_token_times)})")
        print(f"  num_prompt_tokens: {prompt_len}")
        print(f"  num_output_tokens: {tokens_received}")
        print(f"  error_code: {error_response_code}")
        print(f"  error_msg: {error_msg}")

        
        metrics = RequestMetrics(
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        return metrics, generated_text


if __name__ == "__main__":
    # Example usage
    # set api address
    os.environ["OLLAMA_API_BASE"] = "http://4090wsl:11434"

    client = OllamaChatCompletionsClient(
        "phi4:14b-q4_K_M", "huggyllama/llama-7b")
    request_config = RequestConfig(
        prompt=("Hello, how are you?", 1024),
        model="phi4:14b-q4_K_M",
        llm_api="ollama",
        sampling_params={
            "temperature": 0.7,
            "max_tokens": 50,
        },
    )
    metrics, response = client.send_llm_request(request_config)
    print("Response:", response)
    print("Metrics:", metrics)
