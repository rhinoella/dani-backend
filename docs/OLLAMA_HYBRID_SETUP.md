# Ollama Hybrid Setup Guide

DANI Engine now supports both **local Docker Ollama** and **cloud Ollama** with automatic switching via environment variables.

## Quick Start

### Option 1: Local Docker Ollama (Default - Free)

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_API_KEY not needed
```

Start all services including local Ollama:
```bash
docker-compose up
```

### Option 2: Cloud Ollama (Production - Scalable)

```bash
# .env
OLLAMA_BASE_URL=https://api.ollama.ai
OLLAMA_API_KEY=your-cloud-api-key-here
```

Start services without local Ollama:
```bash
docker-compose up qdrant postgres redis api
```

## Environment-Based Configuration

### Development (.env.development)
```bash
OLLAMA_BASE_URL=http://localhost:11434
# No API key needed
```

### Staging (.env.staging)
```bash
OLLAMA_BASE_URL=https://api.ollama.ai
OLLAMA_API_KEY=staging-key-xxx
```

### Production (.env.production)
```bash
OLLAMA_BASE_URL=https://api.ollama.ai
OLLAMA_API_KEY=prod-key-xxx
```

## Docker Compose Usage

### Run with specific environment file:
```bash
docker-compose --env-file .env.production up
```

### Run only specific services:
```bash
# Skip ollama service when using cloud
docker-compose up qdrant postgres redis api
```

## Inside Docker Network

When running API inside Docker, use the service name for local Ollama:
```bash
# docker-compose.yml environment override
OLLAMA_BASE_URL=http://ollama:11434
```

## Benefits by Environment

### Local Docker (Development):
- ✅ Free inference
- ✅ No API key management
- ✅ Works offline
- ✅ Full privacy
- ❌ Slower inference
- ❌ Requires local resources

### Cloud Ollama (Production):
- ✅ Fast professional GPU inference
- ✅ Auto-scaling
- ✅ No local resources needed
- ✅ Always available
- ✅ Automatic model updates
- ❌ Requires internet
- ❌ Pay per use
- ❌ Data leaves your infrastructure

## Testing the Setup

Test local Ollama connection:
```bash
curl http://localhost:11434/api/tags
```

Test cloud Ollama connection:
```bash
curl https://api.ollama.ai/api/tags \
  -H "Authorization: Bearer your-api-key"
```

## Switching Between Environments

Just change the environment variables - no code changes needed:

```bash
# Switch to cloud
export OLLAMA_BASE_URL=https://api.ollama.ai
export OLLAMA_API_KEY=your-key
docker-compose restart api

# Switch back to local
export OLLAMA_BASE_URL=http://localhost:11434
unset OLLAMA_API_KEY
docker-compose up ollama  # if not running
docker-compose restart api
```

## CI/CD Integration

### GitHub Actions Example:
```yaml
env:
  OLLAMA_BASE_URL: https://api.ollama.ai
  OLLAMA_API_KEY: ${{ secrets.OLLAMA_API_KEY }}
```

### GitLab CI Example:
```yaml
variables:
  OLLAMA_BASE_URL: "https://api.ollama.ai"
  OLLAMA_API_KEY: $OLLAMA_API_KEY
```

## Implementation Details

The hybrid support adds only 8 lines of code:
- 1 line in `app/core/config.py` (OLLAMA_API_KEY config)
- 3 lines in `app/llm/ollama.py` (Authorization header)
- 3 lines in `app/embeddings/client.py` (Authorization header)
- 1 line in `.env.example` (documentation)

The clients automatically:
- Include `Authorization: Bearer <key>` header when API key is set
- Work without header when API key is not set (local mode)
- Use same retry logic, circuit breakers, and error handling

## Troubleshooting

### Can't connect to local Ollama:
```bash
# Check if container is running
docker ps | grep ollama

# Check if models are pulled
docker exec dani-ollama ollama list

# Pull required models
docker exec dani-ollama ollama pull qwen2.5:3b
docker exec dani-ollama ollama pull nomic-embed-text
```

### Can't connect to cloud Ollama:
```bash
# Verify API key is set
echo $OLLAMA_API_KEY

# Test authentication
curl https://api.ollama.ai/api/tags \
  -H "Authorization: Bearer $OLLAMA_API_KEY"
```

### Check current configuration:
```bash
python -c "from app.core.config import settings; print(f'URL: {settings.OLLAMA_BASE_URL}'); print(f'Has Key: {bool(settings.OLLAMA_API_KEY)}')"
```
