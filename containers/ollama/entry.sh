#!/bin/bash

echo "Starting Ollama server..."
OLLAMA_MODELS=/models ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 1
done

echo "Creating custom model configuration..."
cat > /Modelfile << EOF
FROM $MODEL

PARAMETER temperature $TEMPERATURE
PARAMETER num_ctx $CONTEXT_SIZE
EOF

echo "Loading model configuration..."
ollama create custom-model -f Modelfile

echo "Model custom-model created successfully."
#ollama run custom-model

wait $SERVE_PID