# Dockerfile
FROM oven/bun:1 AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json bun.lockb ./

# Install dependencies
RUN bun install

# Copy source files
COPY . .

# Ensure the data directory exists
RUN mkdir -p src/data

# Run compression and build steps
RUN NODE_OPTIONS='--max-old-space-size=32768' && \
    echo "Running compression..." && \
    bun run src/compress.ts && \
    echo "Running TypeScript build..." && \
    bun run build:tsc && \
    echo "Running Vite build..." && \
    bun run build:vite

# Production stage
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration if you have any custom settings
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
