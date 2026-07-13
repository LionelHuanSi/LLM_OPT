FROM vllm/vllm-openai:v0.22.1

# Pre-bake model weights into /model directory inside the container
# Ensure Qwen3.5-2B weights exist in local ./model directory before building
COPY model /model

EXPOSE 8000
