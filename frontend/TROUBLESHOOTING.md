# Troubleshooting Guide

## Server Not Starting

### 1. Check if port 5173 is already in use
```powershell
Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue
```

If port is in use, either:
- Stop the process using that port
- Change the port in `vite.config.ts`

### 2. Clear cache and reinstall
```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

### 3. Check Node.js version
```powershell
node --version
```
Should be 18 or higher.

### 4. Run with verbose output
```powershell
npm run dev -- --debug
```

### 5. Check for TypeScript errors
```powershell
npx tsc --noEmit
```

### 6. Check browser console
Open browser DevTools (F12) and check for errors in Console tab.

## Common Issues

### "Cannot find module" errors
- Run `npm install` again
- Delete `node_modules` and reinstall

### "Port already in use"
- Kill the process: `Get-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess | Stop-Process`
- Or change port in `vite.config.ts`

### Blank page / White screen
- Check browser console for errors
- Verify all imports are correct
- Check if `index.html` exists and has correct root div

### API connection errors
- Verify backend services are running
- Check `.env` file has correct URLs
- Check CORS settings on backend

## Manual Start

1. Open PowerShell in the frontend directory
2. Run: `npm run dev`
3. Wait for "Local: http://localhost:5173" message
4. Open that URL in browser

