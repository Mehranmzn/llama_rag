#!/bin/bash
# Script to install Ollama

curl -fsSL https://ollama.com/install.sh | sh


chmod +x install_ollama.sh
./install_ollama.sh

ollama run llama3.1

