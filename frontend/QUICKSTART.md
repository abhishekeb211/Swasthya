# Quick Start Guide

## Prerequisites

- Node.js 18+ installed
- npm, yarn, or pnpm package manager
- Backend services running (orchestrator + agents)

## Setup Steps

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` if your backend URLs differ from defaults.

4. **Start development server:**
   ```bash
   npm run dev
   ```

5. **Open browser:**
   Navigate to `http://localhost:5173`

## Using Mock Data

If backend services are not available, you can modify the API endpoints to return mock data. See `src/lib/api/mock.ts` for sample mock responses.

## Common Issues

### Port Already in Use
If port 5173 is taken, Vite will automatically use the next available port.

### CORS Errors
Ensure your backend services have CORS enabled for `http://localhost:5173`.

### API Connection Failed
- Verify backend services are running
- Check environment variables in `.env`
- Check browser console for detailed error messages

## Building for Production

```bash
npm run build
```

Output will be in the `dist/` directory, ready to be served by any static file server.

## Docker Build

```bash
docker build -t swasthya-frontend .
docker run -p 80:80 swasthya-frontend
```

