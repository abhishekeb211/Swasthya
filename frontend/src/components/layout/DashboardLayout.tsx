import { Sidebar } from './Sidebar'
import { Navbar } from './Navbar'

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden bg-gradient-to-br from-teal-50/50 via-white to-blue-50/50 dark:from-teal-950/50 dark:via-background dark:to-blue-950/50">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto bg-transparent p-4 sm:p-6">{children}</main>
      </div>
    </div>
  )
}

