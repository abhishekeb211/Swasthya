# Project Structure

Complete frontend application for Swasthya Health Intelligence Network.

## Directory Structure

```
frontend/
├── public/                 # Static assets
│   └── vite.svg
├── src/
│   ├── components/        # React components
│   │   ├── ui/           # Base UI components (shadcn/ui style)
│   │   │   ├── badge.tsx
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   └── table.tsx
│   │   ├── layout/       # Layout components
│   │   │   ├── DashboardLayout.tsx
│   │   │   ├── Navbar.tsx
│   │   │   └── Sidebar.tsx
│   │   └── ErrorBoundary.tsx
│   ├── contexts/         # React contexts
│   │   └── ThemeContext.tsx
│   ├── hooks/            # React Query hooks
│   │   ├── index.ts
│   │   ├── useAgentHealth.ts
│   │   ├── useDischarge.ts
│   │   ├── useFL.ts
│   │   ├── useForecast.ts
│   │   ├── useStaff.ts
│   │   └── useTriage.ts
│   ├── lib/              # Utilities and API
│   │   ├── api/
│   │   │   ├── client.ts      # Axios clients
│   │   │   ├── endpoints.ts   # API endpoint definitions
│   │   │   └── mock.ts         # Mock data for development
│   │   └── utils.ts           # Helper functions
│   ├── pages/            # Page components
│   │   ├── Dashboard.tsx
│   │   ├── DemandForecast.tsx
│   │   ├── DischargePlanning.tsx
│   │   ├── ERORScheduling.tsx
│   │   ├── FederatedLearning.tsx
│   │   ├── MLflow.tsx
│   │   ├── StaffScheduling.tsx
│   │   └── Triage.tsx
│   ├── App.tsx           # Main app with routing
│   ├── main.tsx          # Entry point
│   ├── index.css         # Global styles
│   └── vite-env.d.ts     # Vite type definitions
├── .env.example          # Environment variables template
├── .gitignore
├── Dockerfile            # Docker build configuration
├── nginx.conf            # Nginx config for production
├── index.html
├── package.json
├── postcss.config.js
├── QUICKSTART.md         # Quick start guide
├── README.md             # Main documentation
├── tailwind.config.js
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

## Key Features

### Pages
1. **Dashboard** - Overview of all agents and operations
2. **Demand Forecast** - Patient admission predictions
3. **Triage & Acuity** - Patient assessment and ER queue
4. **Staff Scheduling** - AI-optimized staff schedules
5. **ER/OR Scheduling** - Emergency and OR management
6. **Discharge Planning** - Discharge readiness analysis
7. **Federated Learning** - FL server monitoring
8. **MLflow** - Model tracking dashboard

### Components
- **UI Components**: Reusable, accessible components
- **Layout Components**: Sidebar navigation and navbar
- **Error Boundary**: Graceful error handling

### Hooks
- Custom React Query hooks for all API endpoints
- Automatic caching and refetching
- Error handling and loading states

### API Integration
- Separate Axios clients for each service
- Type-safe endpoint definitions
- Mock data support for development

## Technology Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **React Query** - Data fetching
- **Recharts** - Data visualization
- **React Router** - Navigation
- **Axios** - HTTP client

## Development Workflow

1. Install dependencies: `npm install`
2. Configure environment: Copy `.env.example` to `.env`
3. Start dev server: `npm run dev`
4. Build for production: `npm run build`

## Production Deployment

- Build creates optimized bundle in `dist/`
- Docker image available with Nginx
- Static file serving with SPA routing support

