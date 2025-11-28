import { useDischargeAnalysis } from '@/hooks'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { LogOut, CheckCircle, AlertTriangle, XCircle } from 'lucide-react'
import { formatDate } from '@/lib/utils'

export function DischargePlanning() {
  const { data: dischargeData, isLoading } = useDischargeAnalysis()

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'needs_review':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'not_ready':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return null
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready':
        return <Badge variant="success">Ready</Badge>
      case 'needs_review':
        return <Badge variant="warning">Needs Review</Badge>
      case 'not_ready':
        return <Badge variant="destructive">Not Ready</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  const getReadinessColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const readyPatients = dischargeData?.filter((d) => d.status === 'ready') || []
  const needsReview = dischargeData?.filter((d) => d.status === 'needs_review') || []
  const notReady = dischargeData?.filter((d) => d.status === 'not_ready') || []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Discharge Planning</h1>
        <p className="text-muted-foreground">AI-powered discharge readiness analysis</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Ready for Discharge</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{readyPatients.length}</div>
            <p className="text-xs text-muted-foreground">Patients ready to go home</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Needs Review</CardTitle>
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{needsReview.length}</div>
            <p className="text-xs text-muted-foreground">Require medical review</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Not Ready</CardTitle>
            <XCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{notReady.length}</div>
            <p className="text-xs text-muted-foreground">Continue treatment</p>
          </CardContent>
        </Card>
      </div>

      {/* Patient Analysis Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LogOut className="h-5 w-5" />
            Patient Discharge Analysis
          </CardTitle>
          <CardDescription>Comprehensive discharge readiness assessment</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">Loading analysis...</div>
          ) : dischargeData && dischargeData.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient ID</TableHead>
                  <TableHead>Readiness Score</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Estimated Discharge</TableHead>
                  <TableHead>Explanation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dischargeData.map((patient) => (
                  <TableRow key={patient.patient_id}>
                    <TableCell className="font-medium">{patient.patient_id}</TableCell>
                    <TableCell>
                      <span className={`text-lg font-bold ${getReadinessColor(patient.readiness_score)}`}>
                        {patient.readiness_score}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {getStatusIcon(patient.status)}
                        {getStatusBadge(patient.status)}
                      </div>
                    </TableCell>
                    <TableCell>
                      {formatDate(patient.estimated_discharge_date)}
                    </TableCell>
                    <TableCell className="max-w-md">
                      <p className="text-sm text-muted-foreground truncate">
                        {patient.explanation}
                      </p>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              No discharge analysis data available
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed View by Status */}
      {readyPatients.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Ready for Discharge</CardTitle>
            <CardDescription>Patients cleared for discharge</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {readyPatients.map((patient) => (
                <div key={patient.patient_id} className="rounded-lg border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-950">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Patient {patient.patient_id}</p>
                      <p className="text-sm text-muted-foreground">
                        Readiness: {patient.readiness_score}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">Discharge Date</p>
                      <p className="text-sm text-muted-foreground">
                        {formatDate(patient.estimated_discharge_date)}
                      </p>
                    </div>
                  </div>
                  <p className="mt-2 text-sm">{patient.explanation}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

