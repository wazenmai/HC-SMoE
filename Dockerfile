FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt update && apt install -y git wget
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY promptsource promptsource
RUN pip install -e promptsource/
COPY lm-evaluation-harness lm-evaluation-harness
RUN pip install -e lm-evaluation-harness/
COPY . .
