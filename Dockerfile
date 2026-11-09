FROM python:3.14-slim

RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONPATH=/home/app/srv \
    PYTHONUNBUFFERED=1
WORKDIR /home/app/srv

COPY --chown=app deploy/requirements-server.txt requirements.txt
RUN pip install --no-cache-dir --user torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app drowsim/ drowsim/
COPY --chown=app server/ server/
COPY --chown=app dashboard/ dashboard/
COPY --chown=app models/bilstm_attention.pt models/bilstm_attention.pt

EXPOSE 10000
CMD ["sh", "-c", "exec python server/server.py --host 0.0.0.0 --port ${PORT:-10000} --bilstm models/bilstm_attention.pt --demo server/demo_telemetry.jsonl"]
