#!/bin/bash

uvicorn src.api:app --host 0.0.0.0 --port 8000 &

streamlit run src/streamlit_app.py \
  --server.port $PORT \
  --server.address 0.0.0.0
