# genai-engineering — Containerized Data Science Portfolio
FROM python:3.10-slim

WORKDIR /app

# Install system deps for data science
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose all dashboard ports
EXPOSE 8531
EXPOSE 8532
EXPOSE 8533
EXPOSE 8534
EXPOSE 8535
EXPOSE 8536
EXPOSE 8537
EXPOSE 8538

# Default: show available dashboards
CMD ["python", "-c", "\nprint('  streamlit run rag-kb -> http://localhost:8531')\nprint('  streamlit run llm-classify -> http://localhost:8532')\nprint('  streamlit run arxiv-engine -> http://localhost:8533')\nprint('  streamlit run pubmed-engine -> http://localhost:8534')\nprint('  streamlit run clinical-trials -> http://localhost:8535')\nprint('  streamlit run congress-bills -> http://localhost:8536')\nprint('  streamlit run scotus-opinions -> http://localhost:8537')\nprint('  streamlit run mlops-registry -> http://localhost:8538')\n"]
