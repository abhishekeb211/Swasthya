import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { Dashboard } from '@/pages/Dashboard'
import { DemandForecast } from '@/pages/DemandForecast'
import { Triage } from '@/pages/Triage'
import { StaffScheduling } from '@/pages/StaffScheduling'
import { ERORScheduling } from '@/pages/ERORScheduling'
import { DischargePlanning } from '@/pages/DischargePlanning'
import { FederatedLearning } from '@/pages/FederatedLearning'
import { MLflow } from '@/pages/MLflow'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <DashboardLayout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/forecast" element={<DemandForecast />} />
              <Route path="/triage" element={<Triage />} />
              <Route path="/staff" element={<StaffScheduling />} />
              <Route path="/er-or" element={<ERORScheduling />} />
              <Route path="/discharge" element={<DischargePlanning />} />
              <Route path="/fl" element={<FederatedLearning />} />
              <Route path="/mlflow" element={<MLflow />} />
            </Routes>
          </DashboardLayout>
        </BrowserRouter>
        <Toaster position="top-right" />
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App

