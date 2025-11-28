import { useState } from 'react'
import { useERQueue, useAddPatient } from '@/hooks'
import { erOrEndpoints } from '@/lib/api/endpoints'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Calendar, UserPlus, GanttChart } from 'lucide-react'
import { formatDateTime } from '@/lib/utils'
import toast from 'react-hot-toast'

export function ERORScheduling() {
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [acuityLevel, setAcuityLevel] = useState('3')
  const [, setSurgeryFile] = useState<File | null>(null)

  const { data: queue } = useERQueue()
  const addPatientMutation = useAddPatient()
  const queryClient = useQueryClient()

  const { data: orSchedule } = useQuery({
    queryKey: ['or-schedule'],
    queryFn: () => erOrEndpoints.getORSchedule().then((res) => res.data),
  })

  const scheduleORMutation = useMutation({
    mutationFn: (data: { surgeries: Array<{ id: string; duration: number; priority: number }> }) =>
      erOrEndpoints.scheduleOR(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['or-schedule'] })
      toast.success('OR schedule generated')
    },
    onError: () => {
      toast.error('Failed to generate OR schedule')
    },
  })

  const handleAddPatient = () => {
    addPatientMutation.mutate({
      name: patientName,
      age: parseInt(patientAge),
      acuity_level: parseInt(acuityLevel),
      status: 'waiting',
    })
    setPatientName('')
    setPatientAge('')
    setAcuityLevel('3')
  }

  const handleScheduleOR = () => {
    // Mock surgery data - in real app, parse from file
    const surgeries = [
      { id: 's1', duration: 120, priority: 5 },
      { id: 's2', duration: 90, priority: 4 },
      { id: 's3', duration: 180, priority: 3 },
    ]
    scheduleORMutation.mutate({ surgeries })
  }

  const sortedQueue = queue
    ? [...queue].sort((a, b) => b.acuity_level - a.acuity_level)
    : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">ER/OR Scheduling</h1>
        <p className="text-muted-foreground">Manage emergency room and operating room schedules</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserPlus className="h-5 w-5" />
              Add Patient to ER Queue
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Patient Name</Label>
              <Input
                id="name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="John Doe"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  placeholder="45"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="acuity">Acuity Level</Label>
                <Input
                  id="acuity"
                  type="number"
                  min="1"
                  max="5"
                  value={acuityLevel}
                  onChange={(e) => setAcuityLevel(e.target.value)}
                />
              </div>
            </div>
            <Button onClick={handleAddPatient} disabled={!patientName || !patientAge}>
              Add to Queue
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GanttChart className="h-5 w-5" />
              OR Schedule
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="surgeryFile">Upload Surgery List (CSV)</Label>
              <Input
                id="surgeryFile"
                type="file"
                accept=".csv"
                onChange={(e) => setSurgeryFile(e.target.files?.[0] || null)}
              />
            </div>
            <Button
              onClick={handleScheduleOR}
              disabled={scheduleORMutation.isPending}
            >
              {scheduleORMutation.isPending ? 'Scheduling...' : 'Generate OR Schedule'}
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>ER Queue (Prioritized)</CardTitle>
          <CardDescription>Patients sorted by acuity level</CardDescription>
        </CardHeader>
        <CardContent>
          {sortedQueue.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Priority</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Acuity Level</TableHead>
                  <TableHead>Arrival Time</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedQueue.map((patient, idx) => (
                  <TableRow key={patient.id}>
                    <TableCell className="font-bold">#{idx + 1}</TableCell>
                    <TableCell>{patient.name}</TableCell>
                    <TableCell>{patient.age}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          patient.acuity_level >= 4
                            ? 'destructive'
                            : patient.acuity_level === 3
                            ? 'warning'
                            : 'success'
                        }
                      >
                        Level {patient.acuity_level}
                      </Badge>
                    </TableCell>
                    <TableCell>{formatDateTime(patient.arrival_time)}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{patient.status}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              No patients in queue
            </div>
          )}
        </CardContent>
      </Card>

      {orSchedule && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5" />
              Operating Room Schedule
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Array.isArray(orSchedule) ? (
                orSchedule.map((surgery: any, idx: number) => (
                  <div key={idx} className="rounded-lg border p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">Surgery {surgery.id || idx + 1}</p>
                        <p className="text-sm text-muted-foreground">
                          Duration: {surgery.duration} minutes
                        </p>
                      </div>
                      <Badge variant="outline">Priority {surgery.priority}</Badge>
                    </div>
                    {surgery.start_time && (
                      <p className="mt-2 text-sm">
                        Scheduled: {formatDateTime(surgery.start_time)}
                      </p>
                    )}
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center py-8 text-muted-foreground">
                  No OR schedule available
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

