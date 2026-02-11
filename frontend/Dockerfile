FROM python:3.12-slim

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir $(python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
print(' '.join(deps))
")

# Copy source
COPY src/ src/

ENV PYTHONPATH=/app/src

CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
