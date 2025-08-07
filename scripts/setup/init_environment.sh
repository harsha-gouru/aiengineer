#!/bin/bash

# AI Engineering Environment Setup Script
# Purpose: Initialize development environment for AI/MLOps learning

echo "🚀 AI Engineering Environment Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python $(python3 --version) found${NC}"
else
    echo -e "${RED}✗ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install base packages
echo -e "\n${YELLOW}Installing base packages...${NC}"
pip install --upgrade pip -q
pip install -q \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    requests \
    python-dotenv \
    pytest \
    black \
    flake8

echo -e "${GREEN}✓ Base packages installed${NC}"

# Check cloud CLIs
echo -e "\n${YELLOW}Checking cloud CLIs...${NC}"

# AWS CLI
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✓ AWS CLI found${NC}"
else
    echo -e "${YELLOW}⚠ AWS CLI not found. Install: https://aws.amazon.com/cli/${NC}"
fi

# Azure CLI
if command -v az &> /dev/null; then
    echo -e "${GREEN}✓ Azure CLI found${NC}"
else
    echo -e "${YELLOW}⚠ Azure CLI not found. Install: https://docs.microsoft.com/cli/azure/install${NC}"
fi

# GCP CLI
if command -v gcloud &> /dev/null; then
    echo -e "${GREEN}✓ GCP CLI found${NC}"
else
    echo -e "${YELLOW}⚠ GCP CLI not found. Install: https://cloud.google.com/sdk/install${NC}"
fi

# Docker
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
else
    echo -e "${YELLOW}⚠ Docker not found. Install: https://docs.docker.com/get-docker/${NC}"
fi

# Create .env template if not exists
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env template...${NC}"
    cat > .env.example << EOF
# Cloud Credentials (DO NOT COMMIT!)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1

AZURE_SUBSCRIPTION_ID=
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=

GCP_PROJECT_ID=
GCP_SERVICE_ACCOUNT_KEY_PATH=

# API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_TOKEN=

# Project Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✓ .env.example created${NC}"
fi

echo -e "\n${GREEN}✅ Environment setup complete!${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure cloud CLIs (if installed)"
echo "3. Copy .env.example to .env and add your credentials"
echo "4. Start with Project 01 in projects/01-serverless-apis/"

echo -e "\n${GREEN}Happy Learning! 🎓${NC}"
