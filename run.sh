#!/bin/bash
# Makefile-like commands for the micro-loan default risk prediction pipeline

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help command
help() {
    echo -e "${BLUE}Micro-Loan Default Risk Prediction Pipeline${NC}"
    echo ""
    echo "Available commands:"
    echo ""
    echo -e "${GREEN}make install${NC}        : Install dependencies"
    echo -e "${GREEN}make test${NC}           : Run component tests"
    echo -e "${GREEN}make quick-start${NC}    : Run quick start examples"
    echo -e "${GREEN}make run${NC}            : Run full pipeline (no HPO)"
    echo -e "${GREEN}make run-hpo${NC}        : Run full pipeline with HPO"
    echo -e "${GREEN}make clean${NC}          : Clean outputs and cache"
    echo -e "${GREEN}make docs${NC}           : Display documentation"
    echo ""
}

# Install dependencies
install() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Run tests
test() {
    echo -e "${BLUE}Running component tests...${NC}"
    python test_pipeline.py
}

# Quick start
quick_start() {
    echo -e "${BLUE}Running quick start examples...${NC}"
    python -c "from quick_start import run_all_examples; run_all_examples()"
}

# Run full pipeline
run() {
    echo -e "${BLUE}Running full pipeline (no HPO)...${NC}"
    python main.py
}

# Run with HPO
run_hpo() {
    echo -e "${BLUE}Running full pipeline with HPO...${NC}"
    python main.py --run_hpo
}

# Clean outputs
clean() {
    echo -e "${YELLOW}Cleaning outputs...${NC}"
    rm -rf outputs/
    rm -rf __pycache__ src/__pycache__
    find . -type f -name '*.pyc' -delete
    find . -type f -name '.DS_Store' -delete
    echo -e "${GREEN}✓ Clean complete${NC}"
}

# Display documentation
docs() {
    echo -e "${BLUE}Displaying documentation...${NC}"
    python DOCUMENTATION.py
}

# Main script logic
if [ $# -eq 0 ]; then
    help
else
    case "$1" in
        install)    install ;;
        test)       test ;;
        quick-start) quick_start ;;
        run)        run ;;
        run-hpo)    run_hpo ;;
        clean)      clean ;;
        docs)       docs ;;
        help)       help ;;
        *)          echo "Unknown command: $1"; help; exit 1 ;;
    esac
fi
