# TokenLens UI 

## Quick Start

### Option 1: Using Docker

1. Install [Docker](https://www.docker.com/)
2. Run:
   ```bash
   docker run -p 8080:80 ghcr.io/kaiserruben/tokenlense:latest
   ```
3. Open `http://localhost:8080`

### Option 2: Local Development

1. Install [Bun](https://bun.sh):
   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```
   For Windows, first install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) from Microsoft Store

2. Install dependencies and start the server:
   ```bash
   bun install
   bun run dev    # Development server
   # OR
   bun run build  # Production build
   bun run preview # Serve production build
   ```