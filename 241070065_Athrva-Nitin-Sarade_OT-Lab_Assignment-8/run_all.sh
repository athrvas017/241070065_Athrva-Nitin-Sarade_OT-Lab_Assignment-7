#!/usr/bin/env bash
# Master execution shell script for Assignment 8
set -e

echo "Activating virtual environment..."
if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

python run_all.py
