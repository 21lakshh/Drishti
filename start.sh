#!/bin/bash
echo "Starting Model Server spanning on 0.0.0.0:8000..."
uvicorn model_server:app --host 0.0.0.0 --port 8000 &

echo "Waiting a few seconds for the Model Server to initialize..."
sleep 5

echo "Starting Livekit Agent..."
python agent.py dev
