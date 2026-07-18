# ============================================================
# Stage 1: Build the frontend (Vite/React dashboard)
# ============================================================
FROM node:24-bookworm AS builder

WORKDIR /workspace

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace manifests first for better layer caching
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY tsconfig.base.json tsconfig.json ./

# Copy only the packages required for the dashboard build
COPY lib/ lib/
COPY artifacts/dashboard/ artifacts/dashboard/

# Install all dependencies (dev deps needed for the build)
RUN pnpm install --frozen-lockfile

# Run codegen so generated API types are present before building
WORKDIR /workspace/lib/api-spec
RUN pnpm run codegen || true

WORKDIR /workspace

# Build the static frontend assets
ENV BASE_PATH=/
ENV PORT=23183
RUN pnpm --filter @workspace/dashboard run build

# ============================================================
# Stage 2: Serve with nginx
# ============================================================
FROM nginx:alpine AS production

# Remove the default nginx site config
RUN rm /etc/nginx/conf.d/default.conf

# SPA-friendly nginx config:
#  - serve files from the Vite output directory
#  - fall back to index.html for all unknown paths (client-side routing)
#  - gzip static assets
RUN cat > /etc/nginx/conf.d/app.conf << 'EOF'
server {
    listen 8080;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # Serve pre-compressed assets when available
    gzip_static on;

    # Brotli (nginx-module-brotli) is not bundled in the base image;
    # gzip_static covers the common case.

    # Cache fingerprinted assets (Vite adds content hashes to filenames)
    location ~* \.(js|css|woff2?|ttf|svg|png|jpg|ico|webp)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # SPA fallback — all unmatched paths route to index.html
    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF

# Copy the built static files from Stage 1
COPY --from=builder /workspace/artifacts/dashboard/dist/public /usr/share/nginx/html

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]