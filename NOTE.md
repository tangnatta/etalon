Run vllm server (on 4090):
```
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-14B-Instruct --dtype auto --api-key token-abc123 -tp 1
```
