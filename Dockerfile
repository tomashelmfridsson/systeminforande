FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    SYSTEMINFORANDE_ENABLE_AGENTIC_RAG=false \
    SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL=openai/gpt-oss-20b \
    SYSTEMINFORANDE_AGENT1_MODEL=openai/gpt-oss-20b \
    SYSTEMINFORANDE_AGENT2_MODEL=openai/gpt-oss-20b \
    SYSTEMINFORANDE_AGENT3_MODEL=openai/gpt-oss-20b \
    SYSTEMINFORANDE_AGENT_CORRECTION_MODEL=openai/gpt-oss-120b

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        git-lfs \
        procps \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

RUN chown -R user:user /app

COPY --chown=user requirements.txt .

USER user

RUN python -m pip install --upgrade pip \
    && python -m pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
