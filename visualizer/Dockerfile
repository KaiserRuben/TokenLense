FROM node:22-slim

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Expose the default Next.js port
EXPOSE 3000

# Run the Next.js application
CMD ["npm", "start"]