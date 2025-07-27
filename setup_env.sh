#!/bin/bash

# Create conda environment for TTC benchmarking
conda create -n ttc-bench python=3.11 -y
conda activate ttc-bench

# Install dependencies
pip install -r requirements.txt

# Download NLTK data for evaluation
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create .env file template
cat > .env << EOF
GROQ_API_KEY=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
GROQ_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=60
EOF

echo "Environment setup complete! Don't forget to:"
echo "1. Add your GROQ_API_KEY to .env"
echo "2. Activate the environment: conda activate ttc-bench"
