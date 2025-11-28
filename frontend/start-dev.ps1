# PowerShell script to start the development server
Write-Host "Starting Swasthya Frontend Development Server..." -ForegroundColor Green
Write-Host ""

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Start the dev server
Write-Host "Starting Vite dev server on http://localhost:5173" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

npm run dev

