# Swasthya Frontend

Frontend application for Swasthya - India's Decentralized Health Intelligence Network. A modern, production-ready React application built with TypeScript, Vite, and TailwindCSS.

## Features

- 🏥 **Multi-Agent Dashboard** - Real-time monitoring of all AI agents
- 📊 **Demand Forecasting** - Upload data and visualize patient admission predictions
- 🚨 **Triage & Acuity** - Assess patient acuity levels and manage ER queue
- 👥 **Staff Scheduling** - AI-optimized staff scheduling with export capabilities
- 📅 **ER/OR Scheduling** - Manage emergency room and operating room schedules
- 🏃 **Discharge Planning** - AI-powered discharge readiness analysis
- 🧠 **Federated Learning** - Monitor FL rounds across multiple servers
- 📈 **MLflow Integration** - Embedded MLflow dashboard for experiment tracking

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **TailwindCSS** for styling
- **shadcn/ui** inspired components
- **React Query** for data fetching and caching
- **Recharts** for data visualization
- **React Router** for navigation
- **Axios** for API calls

## Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Backend services running (orchestrator and agents)

## Installation

1. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Copy environment variables:

```bash
cp .env.example .env
```

3. Update `.env` with your backend URLs if different from defaults.

## Development

Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

The app will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
# or
yarn build
# or
pnpm build
```

The production build will be in the `dist` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/              # Base UI components (Button, Card, etc.)
│   │   └── layout/          # Layout components (Sidebar, Navbar)
│   ├── pages/               # Page components
│   ├── hooks/               # React Query hooks
│   ├── lib/                 # Utilities and API clients
│   │   ├── api/             # API endpoints and clients
│   │   └── utils.ts         # Helper functions
│   ├── contexts/            # React contexts (Theme)
│   ├── App.tsx              # Main app component with routing
│   └── main.tsx             # Entry point
├── public/                  # Static assets
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

## API Integration

The frontend integrates with:

- **Orchestrator** (port 3000): Main API gateway
- **Forecast Agent** (port 8001): Demand forecasting
- **Staff Agent** (port 8002): Staff scheduling
- **ER/OR Agent** (port 8003): Emergency and OR scheduling
- **Discharge Agent** (port 8004): Discharge planning
- **Triage Agent** (port 8005): Triage and acuity assessment
- **FL Servers** (ports 8086, 8087): Federated learning
- **MLflow** (port 5000): Model tracking

## Environment Variables

All API URLs can be configured via environment variables:

- `VITE_ORCHESTRATOR_URL` - Orchestrator API URL
- `VITE_FORECAST_AGENT_URL` - Forecast agent URL
- `VITE_STAFF_AGENT_URL` - Staff agent URL
- `VITE_ER_OR_AGENT_URL` - ER/OR agent URL
- `VITE_DISCHARGE_AGENT_URL` - Discharge agent URL
- `VITE_TRIAGE_AGENT_URL` - Triage agent URL
- `VITE_FL_SERVER_1_URL` - FL Server 1 URL
- `VITE_FL_SERVER_2_URL` - FL Server 2 URL
- `VITE_MLFLOW_URL` - MLflow dashboard URL

## Features by Page

### Dashboard
- Real-time agent health monitoring
- Today's forecast summary
- ER queue status
- Discharge readiness overview
- Interactive charts

### Demand Forecast
- CSV data upload for training
- Configurable forecast periods (7/14/30 days)
- Interactive line charts with confidence intervals
- Forecast summary statistics

### Triage & Acuity
- Patient assessment form (symptoms, vitals, lab readings)
- AI-powered acuity level determination
- ER queue management
- Next patient retrieval

### Staff Scheduling
- Staff member listing
- AI-optimized schedule generation
- Schedule visualization
- CSV export functionality

### ER/OR Scheduling
- Add patients to ER queue
- Prioritized patient list
- OR schedule generation
- Surgery list upload

### Discharge Planning
- Comprehensive discharge analysis
- Readiness score visualization
- Color-coded status indicators
- Estimated discharge dates

### Federated Learning
- Dual server monitoring
- Round status and metrics
- Client participation tracking
- Training history visualization

### MLflow
- Embedded MLflow dashboard
- Experiment tracking
- Model registry access

## Styling

The app uses TailwindCSS with a custom healthcare theme. Colors are configured in `tailwind.config.js` with teal/blue accents suitable for healthcare applications.

Dark mode is supported and can be toggled via the theme switcher in the navbar.

## Error Handling

- React Query handles API errors automatically
- Error boundaries catch component errors
- Toast notifications for user feedback
- Loading states for all async operations

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

Part of the Swasthya project.

