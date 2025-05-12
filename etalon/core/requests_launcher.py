from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from threading import Thread
from typing import Dict, List

from etalon.config.config import ClientConfig
from etalon.core.llm_clients import construct_client
from etalon.core.llm_clients.base_llm_client import BaseLLMClient


def run_client_process(
    client_id: int,
    client_config_dict: dict,
    input_queue: MPQueue,
    output_queue: MPQueue,
) -> None:
    """Run the client in a separate process."""
    from etalon.config.config import ClientConfig

    # Reconstruct the client config from the dictionary, excluding non-constructor attributes
    # Only include fields that are defined in the ClientConfig's __init__
    init_args = {
        "model": client_config_dict.get("model", "gpt-3.5-turbo"),
        "tokenizer": client_config_dict.get("tokenizer"),
        "num_clients": client_config_dict.get("num_clients", 2),
        "num_concurrent_requests_per_client": client_config_dict.get("num_concurrent_requests_per_client", 5),
        "additional_sampling_params": client_config_dict.get("additional_sampling_params", "{}"),
        "llm_api": client_config_dict.get("llm_api", "openai"),
        "address_append_value": client_config_dict.get("address_append_value", "chat/completions"),
    }

    client_config = ClientConfig(**init_args)

    # Create the client inside the process
    assert client_config.tokenizer is not None
    assert client_config.model is not None

    llm_client = construct_client(
        model_name=client_config.model,
        tokenizer_name=client_config.tokenizer,
        llm_api=client_config.llm_api,
    )

    # Start the client threads
    client_threads = [
        Thread(target=process_requests, args=(
            llm_client, input_queue, output_queue))
        for _ in range(client_config.num_concurrent_requests_per_client)
    ]

    for thread in client_threads:
        thread.start()

    for thread in client_threads:
        thread.join()


def process_requests(
    llm_client: BaseLLMClient,
    input_queue: MPQueue,
    output_queue: MPQueue,
) -> None:
    """Process requests from the input queue."""
    while True:
        request_config = input_queue.get()
        if request_config is None:
            break
        result = llm_client.send_llm_request(request_config)
        output_queue.put(result)


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.clients: List[Process] = []
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Create a clean dictionary with only the initialization parameters
        client_config_dict = {
            "model": client_config.model,
            "tokenizer": client_config.tokenizer,
            "num_clients": client_config.num_clients,
            "num_concurrent_requests_per_client": client_config.num_concurrent_requests_per_client,
            "additional_sampling_params": client_config.additional_sampling_params,
            "llm_api": client_config.llm_api,
            "address_append_value": client_config.address_append_value,
        }

        for client_id in range(self.client_config.num_clients):
            client = Process(
                target=run_client_process,
                args=(
                    client_id,
                    client_config_dict,
                    input_queue,
                    output_queue,
                ),
            )
            self.clients.append(client)

    def start(self) -> None:
        """Start the clients."""
        for client in self.clients:
            client.start()

    def complete_tasks(self) -> None:
        """Complete the clients."""
        # put None to indicate that client should stop
        for _ in range(
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        ):
            self.input_queue.put(None)

        for client in self.clients:
            client.join()

    def kill_clients(self) -> None:
        """Kill all the clients."""
        for client in self.clients:
            client.terminate()
            client.join(30)
            client.kill()
            client.close()
