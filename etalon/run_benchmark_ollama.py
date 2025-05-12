import multiprocessing
import os
import platform
import random
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import Any, List, Optional

from tqdm import tqdm

from etalon.config.config import BenchmarkConfig, ClientConfig
from etalon.core.hf_utils import get_tokenizer
from etalon.core.request_config import RequestConfig
from etalon.core.requests_launcher import RequestsLauncher
from etalon.logger import init_logger
from etalon.metrics.service_metrics import ServiceMetrics
from etalon.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from etalon.request_generator.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from etalon.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from etalon.request_generator.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from etalon.request_generator.utils import generate_random_prompt
from etalon.config.config import (
    GammaRequestIntervalGeneratorConfig,
    ZipfRequestLengthGeneratorConfig,
    MetricsConfig,)

logger = init_logger(__name__)


def get_request_params(
    client_config: ClientConfig,
    tokenizer: Any,
    request_length_generator: BaseRequestLengthGenerator,
    corpus_lines: Optional[List[str]] = None,
    request_id: Optional[int] = None,
) -> RequestConfig:
    (
        num_prompt_tokens,
        num_output_tokens,
    ) = request_length_generator.get_next_num_tokens()
    if num_prompt_tokens < 0 or num_output_tokens < 0:
        logger.error(
            f"Invalid number of tokens generated: prompt={num_prompt_tokens}, output={num_output_tokens} (potentially from trace request length generator)."
        )
    num_prompt_tokens = int(num_prompt_tokens)
    num_output_tokens = int(num_output_tokens)
    prompt = generate_random_prompt(
        tokenizer=tokenizer,
        num_prompt_tokens=num_prompt_tokens,
        num_output_tokens=num_output_tokens,
        corpus_lines=corpus_lines,
    )
    default_sampling_params = {"max_tokens": num_output_tokens}
    default_sampling_params.update(
        client_config.additional_sampling_params_dict)
    request_config = RequestConfig(
        model=client_config.model,
        prompt=prompt,
        sampling_params=default_sampling_params,
        llm_api="ollama",  # Always use ollama for this script
        address_append_value=client_config.address_append_value,
        id=request_id,
    )

    return request_config


def should_send_new_request(
    service_metrics: ServiceMetrics, num_errored_requests_handled: int
) -> bool:
    """Check if a request should be sent based on the current state of the service."""
    return (service_metrics.num_requests < service_metrics.max_requests) or (
        service_metrics.num_requests >= service_metrics.max_requests
        and num_errored_requests_handled < service_metrics.num_errored_requests
    )


def dispatch_requests(
    input_queue: Queue,
    service_metrics: ServiceMetrics,
    client_config: ClientConfig,
    tokenizer: Any,
    requests_interval_generator: BaseRequestIntervalGenerator,
    requests_length_generator: BaseRequestLengthGenerator,
    corpus_lines: List[str],
    stop_event: threading.Event,
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # Check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1

            # Create and dispatch request
            service_metrics.register_launched_request()
            request_config = get_request_params(
                client_config=client_config,
                tokenizer=tokenizer,
                request_length_generator=requests_length_generator,
                corpus_lines=corpus_lines.copy(),
                request_id=service_metrics.num_requests,
            )
            input_queue.put(request_config)

            # Wait for next interval
            next_request_interval = (
                requests_interval_generator.get_next_inter_request_time()
            )

            if next_request_interval < 0:
                logger.warning(
                    f"Invalid interval {next_request_interval} (potentially from trace interval generator). Stopping the main loop."
                )
                break

            while not stop_event.is_set():
                if time.monotonic() - request_start_time >= next_request_interval:
                    break
                time.sleep(0.01)
        else:
            time.sleep(0.01)


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_texts: List[str],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    while not stop_event.is_set() or not output_queue.empty():
        try:
            result = output_queue.get(timeout=0.1)
            request_metrics, generated_text = result
            if generated_text:
                service_metrics.add_request_metrics(request_metrics)
                generated_texts.append(generated_text)

            pbar.update(service_metrics.num_completed_requests - pbar.n)
        except Empty:
            continue


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    requests_interval_generator: BaseRequestIntervalGenerator,
    requests_length_generator: BaseRequestLengthGenerator,
    service_metrics: ServiceMetrics,
    corpus_lines: List[str],
    generated_texts: List[str],
    pbar: tqdm,
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")

    assert (
        benchmark_config.client_config.tokenizer is not None
    ), "Tokenizer is required."
    tokenizer = get_tokenizer(
        tokenizer_name=benchmark_config.client_config.tokenizer,
        trust_remote_code=True,
    )

    # Create queues for communication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()

    # Initialize request launcher
    req_launcher = RequestsLauncher(
        client_config=benchmark_config.client_config,
        input_queue=input_queue,
        output_queue=output_queue,
    )

    # Start the request launcher processes
    req_launcher.start()

    # Create and start producer-consumer threads
    dispatcher_thread = Thread(
        target=dispatch_requests,
        args=(
            input_queue,
            service_metrics,
            benchmark_config.client_config,
            tokenizer,
            requests_interval_generator,
            requests_length_generator,
            corpus_lines,
            stop_event,
        ),
    )

    processor_thread = Thread(
        target=process_results,
        args=(
            output_queue,
            service_metrics,
            generated_texts,
            pbar,
            stop_event,
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")

    # Signal threads to stop and wait for completion
    stop_event.set()
    dispatcher_thread.join()
    processor_thread.join()

    # Terminate all clients
    req_launcher.kill_clients()

    pbar.close()
    logger.info("Main loop completed.")


def run_benchmark(
    benchmark_config: BenchmarkConfig = None,
):
    """Get the token throughput and latencies for the given model using Ollama API.

    Args:
        benchmark_config: The benchmark configuration. If None, a default config for Ollama will be created.

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    # Set Ollama API address
    os.environ["OLLAMA_API_BASE"] = "https://x8dmhtl5-11434.asse.devtunnels.ms/"

    # Create default config if none provided
    if benchmark_config is None:
        client_config = ClientConfig(
            model="llama2",  # Default model
            tokenizer="hf/TinyLlama/TinyLlama-1.1B-Chat",  # Default tokenizer
            llm_api="ollama",  # Always use Ollama
        )
        benchmark_config = BenchmarkConfig(
            client_config=client_config,
            max_completed_requests=100,  # Default to 100 requests
            timeout=300,  # Default timeout of 5 minutes
        )
    else:
        # Override the llm_api to ensure using Ollama
        benchmark_config.client_config.llm_api = "ollama"

    service_metrics = ServiceMetrics(
        max_requests=benchmark_config.max_completed_requests,
        timeout=benchmark_config.timeout,
        deadline_config=benchmark_config.deadline_config,
        metrics_config=benchmark_config.metrics_config,
        prefill_profiler_config=benchmark_config.prefill_profiler_config,
    )

    generated_texts = []
    pbar = tqdm(total=benchmark_config.max_completed_requests)

    requests_interval_generator = RequestIntervalGeneratorRegistry.get(
        benchmark_config.request_interval_generator_config.get_type(),
        benchmark_config.request_interval_generator_config,
    )
    requests_length_generator = RequestLengthGeneratorRegistry.get(
        benchmark_config.request_length_generator_config.get_type(),
        benchmark_config.request_length_generator_config,
    )

    corpus_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    )
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    run_main_loop(
        benchmark_config=benchmark_config,
        requests_interval_generator=requests_interval_generator,
        requests_length_generator=requests_length_generator,
        service_metrics=service_metrics,
        corpus_lines=corpus_lines,
        generated_texts=generated_texts,
        pbar=pbar,
    )

    logger.info(
        f"Results for token benchmark for {benchmark_config.client_config.model} queried with the Ollama API. {service_metrics}"
    )

    print("Storing metrics...")
    service_metrics.store_output()
    print("Done storing metrics.")
    logger.info(f"Metrics stored to {service_metrics.output_dir}")

    # store the generated texts
    with open(
        os.path.join(service_metrics.output_dir,
                     "generated_texts.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(("\n" + "-" * 30 + "\n").join(generated_texts))


if __name__ == "__main__":
    # On Windows, we need to use 'spawn' method (default), but ensure we execute after if __name__ == '__main__'
    # On MacOS, use 'fork' method
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    # try:
    #     # Try to parse CLI ags first
    #     benchmark_config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    #     # Override the LLM API to ensure we use Ollama
    #     benchmark_config.client_config.llm_api = "ollama"
    # except Exception as e:
    #     logger.warning(
    #         f"Failed to parse CLI args: {e}. Using default Ollama configuration.")
    #     benchmark_config = None

    os.environ["OLLAMA_API_BASE"] = "http://4090wsl:11434"

    # List of models to benchmark
    models_to_test = [
        "phi4:14b-q4_K_M",
        "qwen2.5:14b-instruct-q4_K_M",
        "mistral-nemo:12b-instruct-2407-q4_K_M",
        "gemma3:12b-it-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M"
    ]

    for model in models_to_test:
        logger.info(f"Starting benchmark for model: {model}")

        # Create output directory based on model name
        output_dir = f"{model.split(':')[0]}_results_r300_t1000_cl2_con3"

        # Create benchmark config from specified parameters
        benchmark_config = BenchmarkConfig(
            client_config=ClientConfig(
                model=model,
                tokenizer="huggyllama/llama-7b",
                num_clients=2,
                num_concurrent_requests_per_client=3,
                llm_api="ollama",
            ),
            request_interval_generator_config=GammaRequestIntervalGeneratorConfig(),
            request_length_generator_config=ZipfRequestLengthGeneratorConfig(
                max_tokens=8192),
            metrics_config=MetricsConfig(output_dir=output_dir),
            max_completed_requests=300,
            timeout=1000,
        )

        random.seed(benchmark_config.seed)  # Default seed for reproducibility
        run_benchmark(benchmark_config=benchmark_config)

        logger.info(f"Completed benchmark for model: {model}")
        # Add a small delay between models to ensure proper cleanup
        time.sleep(5)
