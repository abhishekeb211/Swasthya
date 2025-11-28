import { useAgentHealth, useForecast, useERQueue, useDischargeAnalysis } from '@/hooks'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Activity,
  TrendingUp,
  LogOut,
  AlertCircle,
  Users,
  Heart,
  Clock,
  CheckCircle2,
} from 'lucide-react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

const COLORS = ['#14b8a6', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981']

export function Dashboard() {
  const { data: agentHealth, isLoading: healthLoading } = useAgentHealth()
  const { data: forecast, isLoading: forecastLoading } = useForecast(7)
  const { data: erQueue, isLoading: queueLoading } = useERQueue()
  const { data: dischargeData, isLoading: dischargeLoading } = useDischargeAnalysis()

  const readyForDischarge = dischargeData?.filter((d) => d.status === 'ready').length || 0
  const needsReview = dischargeData?.filter((d) => d.status === 'needs_review').length || 0
  const notReady = dischargeData?.filter((d) => d.status === 'not_ready').length || 0

  const healthyAgents = agentHealth?.filter((a) => a.status === 'healthy').length || 0
  const totalAgents = agentHealth?.length || 0

  // Prepare data for charts
  const dischargePieData = [
    { name: 'Ready', value: readyForDischarge, color: '#10b981' },
    { name: 'Needs Review', value: needsReview, color: '#f59e0b' },
    { name: 'Not Ready', value: notReady, color: '#ef4444' },
  ]

  const acuityDistribution = erQueue?.reduce(
    (acc, patient) => {
      const level = patient.acuity_level
      acc[level] = (acc[level] || 0) + 1
      return acc
    },
    {} as Record<number, number>
  ) || {}

  const acuityBarData = Object.entries(acuityDistribution).map(([level, count]) => ({
    level: `Level ${level}`,
    patients: count,
  }))

  const agentPerformanceData = agentHealth?.map((agent) => ({
    name: agent.agent.split(' ')[0],
    responseTime: agent.responseTime || 0,
    status: agent.status === 'healthy' ? 100 : 0,
  })) || []

  return (
    <div className="space-y-6 p-4 sm:p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-teal-600 to-blue-600 bg-clip-text text-transparent">
          Dashboard
        </h1>
        <p className="text-muted-foreground mt-2">Real-time overview of hospital operations and AI agents</p>
      </div>

      {/* Stats Grid - Colorful Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-2 border-teal-200 bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-950 dark:to-teal-900 dark:border-teal-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-teal-900 dark:text-teal-100">
              Agent Health
            </CardTitle>
            <div className="h-10 w-10 rounded-full bg-teal-500 flex items-center justify-center">
              <Activity className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-teal-700 dark:text-teal-300">
              {healthLoading ? '...' : `${healthyAgents}/${totalAgents}`}
            </div>
            <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">Agents operational</p>
            <div className="mt-2 h-2 bg-teal-200 dark:bg-teal-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all"
                style={{ width: `${(healthyAgents / totalAgents) * 100}%` }}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-red-200 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-950 dark:to-red-900 dark:border-red-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-red-900 dark:text-red-100">ER Queue</CardTitle>
            <div className="h-10 w-10 rounded-full bg-red-500 flex items-center justify-center">
              <AlertCircle className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-700 dark:text-red-300">
              {queueLoading ? '...' : erQueue?.length || 0}
            </div>
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">Patients waiting</p>
            <div className="flex items-center gap-1 mt-2">
              <Clock className="h-3 w-3 text-red-500" />
              <span className="text-xs text-red-600 dark:text-red-400">Avg wait: 15 min</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 dark:border-green-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-green-900 dark:text-green-100">
              Ready for Discharge
            </CardTitle>
            <div className="h-10 w-10 rounded-full bg-green-500 flex items-center justify-center">
              <CheckCircle2 className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-700 dark:text-green-300">
              {dischargeLoading ? '...' : readyForDischarge}
            </div>
            <p className="text-xs text-green-600 dark:text-green-400 mt-1">Patients ready</p>
            <div className="flex items-center gap-1 mt-2">
              <TrendingUp className="h-3 w-3 text-green-500" />
              <span className="text-xs text-green-600 dark:text-green-400">+12% this week</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 dark:border-blue-800">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-blue-900 dark:text-blue-100">
              7-Day Forecast
            </CardTitle>
            <div className="h-10 w-10 rounded-full bg-blue-500 flex items-center justify-center">
              <TrendingUp className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">
              {forecastLoading ? '...' : forecast?.[forecast.length - 1]?.predicted?.toFixed(0) || 0}
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">Expected patients</p>
            <div className="flex items-center gap-1 mt-2">
              <Heart className="h-3 w-3 text-blue-500" />
              <span className="text-xs text-blue-600 dark:text-blue-400">Peak: Tomorrow</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="border-2 border-teal-200 dark:border-teal-800">
          <CardHeader className="bg-gradient-to-r from-teal-500 to-blue-500 text-white rounded-t-lg">
            <CardTitle className="text-white">7-Day Demand Forecast</CardTitle>
            <CardDescription className="text-teal-100">Predicted patient admissions with confidence intervals</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {forecastLoading ? (
              <div className="flex h-[300px] items-center justify-center">Loading...</div>
            ) : forecast && forecast.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={forecast}>
                  <defs>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorUpper" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                    stroke="#6b7280"
                  />
                  <YAxis stroke="#6b7280" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="upper_bound"
                    stroke="#10b981"
                    fillOpacity={1}
                    fill="url(#colorUpper)"
                    name="Upper Confidence"
                  />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    stroke="#14b8a6"
                    strokeWidth={3}
                    fillOpacity={1}
                    fill="url(#colorPredicted)"
                    name="Predicted"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower_bound"
                    stroke="#f59e0b"
                    fillOpacity={0.3}
                    fill="#fef3c7"
                    name="Lower Confidence"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No forecast data available
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-2 border-purple-200 dark:border-purple-800">
          <CardHeader className="bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-t-lg">
            <CardTitle className="text-white">Discharge Status Distribution</CardTitle>
            <CardDescription className="text-purple-100">Patient discharge readiness breakdown</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {dischargeLoading ? (
              <div className="flex h-[300px] items-center justify-center">Loading...</div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={dischargePieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {dischargePieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="border-2 border-orange-200 dark:border-orange-800">
          <CardHeader className="bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-t-lg">
            <CardTitle className="text-white">ER Acuity Level Distribution</CardTitle>
            <CardDescription className="text-orange-100">Patient acuity levels in emergency queue</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {queueLoading ? (
              <div className="flex h-[300px] items-center justify-center">Loading...</div>
            ) : acuityBarData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={acuityBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="level" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="patients" fill="#f59e0b" radius={[8, 8, 0, 0]}>
                    {acuityBarData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No ER queue data available
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-2 border-indigo-200 dark:border-indigo-800">
          <CardHeader className="bg-gradient-to-r from-indigo-500 to-blue-500 text-white rounded-t-lg">
            <CardTitle className="text-white">Agent Performance</CardTitle>
            <CardDescription className="text-indigo-100">Response times and health status</CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {healthLoading ? (
              <div className="flex h-[300px] items-center justify-center">Loading...</div>
            ) : agentPerformanceData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={agentPerformanceData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" stroke="#6b7280" />
                  <YAxis dataKey="name" type="category" stroke="#6b7280" width={80} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="responseTime" fill="#3b82f6" radius={[0, 8, 8, 0]}>
                    {agentPerformanceData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.status === 100 ? '#10b981' : '#ef4444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No agent data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Agent Health Status - Enhanced */}
      <Card className="border-2 border-cyan-200 dark:border-cyan-800">
        <CardHeader className="bg-gradient-to-r from-cyan-500 to-teal-500 text-white rounded-t-lg">
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Agent Health Status
          </CardTitle>
          <CardDescription className="text-cyan-100">Real-time status of all AI agents</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          {healthLoading ? (
            <div className="flex items-center justify-center py-8">Loading...</div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
              {agentHealth?.map((agent, index) => (
                <div
                  key={agent.agent}
                  className={`rounded-xl p-4 border-2 transition-all hover:scale-105 ${
                    agent.status === 'healthy'
                      ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-300 dark:from-green-950 dark:to-emerald-950 dark:border-green-700'
                      : 'bg-gradient-to-br from-red-50 to-rose-50 border-red-300 dark:from-red-950 dark:to-rose-950 dark:border-red-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div
                      className={`h-3 w-3 rounded-full ${
                        agent.status === 'healthy' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                      }`}
                    />
                    <Badge
                      variant={agent.status === 'healthy' ? 'success' : 'destructive'}
                      className="text-xs"
                    >
                      {agent.status}
                    </Badge>
                  </div>
                  <p className="font-semibold text-sm mb-1">{agent.agent}</p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    <span>{agent.responseTime}ms</span>
                  </div>
                  <div className="mt-3 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        agent.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                      }`}
                      style={{ width: agent.status === 'healthy' ? '100%' : '0%' }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
