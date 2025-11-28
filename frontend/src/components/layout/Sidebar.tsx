import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  TrendingUp,
  Users,
  Calendar,
  Stethoscope,
  LogOut,
  Activity,
  Brain,
  Database,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Demand Forecast', href: '/forecast', icon: TrendingUp },
  { name: 'Triage & Acuity', href: '/triage', icon: Stethoscope },
  { name: 'Staff Scheduling', href: '/staff', icon: Users },
  { name: 'ER/OR Scheduling', href: '/er-or', icon: Calendar },
  { name: 'Discharge Planning', href: '/discharge', icon: LogOut },
  { name: 'Federated Learning', href: '/fl', icon: Brain },
  { name: 'MLflow', href: '/mlflow', icon: Database },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <div className="flex h-screen w-64 flex-col border-r-2 border-teal-200 dark:border-teal-800 bg-gradient-to-b from-teal-50 to-white dark:from-teal-950 dark:to-background">
      <div className="flex h-16 items-center border-b-2 border-teal-200 dark:border-teal-800 px-6 bg-gradient-to-r from-teal-500 to-blue-500">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-white/20 flex items-center justify-center">
            <Activity className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold text-white">Swasthya</span>
        </div>
      </div>
      <nav className="flex-1 space-y-2 px-3 py-4 overflow-y-auto">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg shadow-teal-500/50 scale-105'
                  : 'text-muted-foreground hover:bg-gradient-to-r hover:from-teal-100 hover:to-blue-100 dark:hover:from-teal-900 dark:hover:to-blue-900 hover:text-foreground hover:scale-105'
              )}
            >
              <item.icon className={cn('h-5 w-5', isActive ? 'text-white' : 'text-teal-600 dark:text-teal-400')} />
              {item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}

