FROM node:20-alpine AS build
WORKDIR /app

COPY frontend/package*.json /app/
RUN npm ci

COPY frontend /app

ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
