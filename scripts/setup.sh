#!/usr/bin/env bash
# scripts/setup.sh
# One-time project bootstrap. Run once after cloning.
# Usage: bash scripts/setup.sh

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Vision-RAG Setup ===${NC}"

# 1. Python venv
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}Creating virtual environment...${NC}"
  python3 -m venv venv
fi
source venv/bin/activate

# 2. Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}Dependencies installed.${NC}"

# 3. .env setup
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo -e "${YELLOW}.env created from .env.example — fill in DB credentials.${NC}"
fi

# 4. Directories
mkdir -p data/raw data/processed/crops data/chroma_db models logs

# 5. Django migrations
echo -e "${YELLOW}Running Django migrations...${NC}"
export DJANGO_SETTINGS_MODULE=api.settings
python manage.py makemigrations api --no-input 2>/dev/null || true
python manage.py migrate --no-input

# 6. Superuser hint
echo -e "${YELLOW}Create a Django superuser (for API token):${NC}"
echo "  python manage.py createsuperuser"
echo "  Then: python manage.py drf_create_token <username>"

# 7. Ollama models
echo -e "${YELLOW}Pulling Ollama models (requires Ollama to be running)...${NC}"
ollama pull llama3 2>/dev/null && echo "  llama3 ready" || echo "  Ollama not running — pull manually: ollama pull llama3"
ollama pull qwen2.5vl 2>/dev/null && echo "  qwen2.5vl ready" || echo "  Pull manually: ollama pull qwen2.5vl"

echo -e "${GREEN}=== Setup complete! ===${NC}"
echo ""
echo "Start the project:"
echo "  1. source venv/bin/activate"
echo "  2. ollama serve                          # terminal 1"
echo "  3. python manage.py runserver            # terminal 2"
echo "  4. streamlit run streamlit_app/app.py    # terminal 3"
