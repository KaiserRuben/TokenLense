# 🔍 TokenLens UI

[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/kaiserruben/tokenlense)
[![Runtime](https://img.shields.io/badge/Runtime-Bun-black)](https://bun.sh)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)

A token relationship visualization framework for analyzing Large Language Model (LLM) token associations and importance metrics. TokenLens provides quantitative insights into token-level interactions and relationships within LLM inputs and outputs.

## 🔧 Technical Overview

TokenLens implements interactive visualizations for:
- Token-to-token relationship matrices
- Association strength metrics
- Normalized importance values

## 🚀 Deployment Options

### 1. Docker Quick Start (Pre-loaded Data)
```bash
docker run -p 8080:80 ghcr.io/kaiserruben/tokenlense:latest
```
> ℹ️ Docker image includes ~50 example datasets for immediate exploration

### 2. Local Development (Custom Data)
Prerequisites: [Bun Runtime](https://bun.sh), [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows only)

```bash
bun install
bun run dev    # Development server
# OR
bun run build  # Production build
bun run preview # Serve production build
```

> ⚠️ To use custom JSON files:
> 1. Place your files in `src/data/` directory
> 2. Rebuild the application

## 📊 Data Structure Specification

TokenLens processes JSON files containing token analysis data. Reference implementation at `src/data/sample.json`:

```json
{
  "metadata": {
    "timestamp": "ISO-8601 timestamp",
    "llm_id": "model identifier",
    "llm_version": "version string",
    "prompt": "input prompt",
    "generation_params": {
      "max_new_tokens": "integer",
      "temperature": "float",
      "top_p": "float",
      "repetition_penalty": "float"
    }
  },
  "data": {
    "input_tokens": [
      {
        "token": "string",
        "token_id": "integer",
        "clean_token": "string"
      }
    ],
    "output_tokens": ["token objects"],
    "association_matrix": "2D float array",
    "normalized_association": "2D float array"
  }
}
```

## ⚙️ Visualization Parameters

### Token Relationship Analysis
- Maximum connections per token: Integer
- Background visibility: Boolean
- Connection visibility: Boolean

### Matrix Visualization
- Association strength rendering
- Normalized importance metrics

## 🗂️ Data Management

### Pre-loaded Data (Docker)
- Access ~50 example visualizations
- Ideal for exploration and testing
- No configuration required

### Custom Data Usage
1. Local Development:
   - Add JSON files to `src/data/`
   - Files must follow specification above
   - Rebuild application

2. Docker with Custom Data:
   ```bash
   # 1. Clone repository
   # 2. Add custom JSON files to src/data/
   # 3. Build custom Docker image
   docker build -t tokenlens-custom .
   # 4. Run custom image
   docker run -p 8080:80 tokenlens-custom
   ```