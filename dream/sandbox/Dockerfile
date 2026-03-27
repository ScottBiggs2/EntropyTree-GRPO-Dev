# Minimal Python sandbox for GRPO code execution (see dream/PLAN_03_ENVIRONMENT_SCALEUP.md Step 1).
FROM python:3.11-slim

COPY entrypoint.py /entrypoint.py
RUN chmod +x /entrypoint.py

ENTRYPOINT ["python", "-u", "/entrypoint.py"]
