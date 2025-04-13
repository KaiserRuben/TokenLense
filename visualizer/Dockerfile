FROM node:22-slim

WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_DISABLE_CSS_OPTIMIZATION=1

# Install dependencies
COPY package.json package-lock.json* ./
# Fixes a bug with lightningcss (tailwind 4)
RUN npm install lightningcss-linux-x64-gnu@1.29.3
RUN npm install @tailwindcss/oxide-linux-x64-gnu
RUN npm ci

# Copy the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the default Next.js port
EXPOSE 3000

# Run the Next.js application
CMD ["npm", "start"]