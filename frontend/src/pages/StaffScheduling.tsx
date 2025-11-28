import { useState } from 'react'
import { useStaff, useGenerateSchedule, useSchedule } from '@/hooks'
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
import { Users, Calendar, Download } from 'lucide-react'
import { formatDate } from '@/lib/utils'

export function StaffScheduling() {
  const [startDate, setStartDate] = useState(
    new Date().toISOString().split('T')[0]
  )
  const [endDate, setEndDate] = useState(
    new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
  )

  const { data: staff, isLoading: staffLoading } = useStaff()
  const { data: schedule, isLoading: scheduleLoading } = useSchedule(startDate, endDate)
  const generateMutation = useGenerateSchedule()

  const handleGenerate = () => {
    generateMutation.mutate({ start_date: startDate, end_date: endDate })
  }

  const handleExport = () => {
    if (!schedule) return

    const csv = [
      ['Staff ID', 'Name', 'Date', 'Start Time', 'End Time'].join(','),
      ...schedule.flatMap((s) =>
        s.shifts.map((shift) =>
          [
            s.staff_id,
            staff?.find((st) => st.id === s.staff_id)?.name || 'Unknown',
            shift.date,
            shift.start_time,
            shift.end_time,
          ].join(',')
        )
      ),
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `schedule-${startDate}-${endDate}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Staff Scheduling</h1>
        <p className="text-muted-foreground">Optimize staff schedules using AI</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Schedule Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="startDate">Start Date</Label>
              <Input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="endDate">End Date</Label>
              <Input
                id="endDate"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button onClick={handleGenerate} disabled={generateMutation.isPending}>
                {generateMutation.isPending ? 'Generating...' : 'Generate Schedule'}
              </Button>
              {schedule && schedule.length > 0 && (
                <Button onClick={handleExport} variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Staff Members
            </CardTitle>
          </CardHeader>
          <CardContent>
            {staffLoading ? (
              <div className="flex items-center justify-center py-8">Loading...</div>
            ) : staff && staff.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Department</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {staff.map((member) => (
                    <TableRow key={member.id}>
                      <TableCell className="font-medium">{member.name}</TableCell>
                      <TableCell>{member.role}</TableCell>
                      <TableCell>{member.department}</TableCell>
                      <TableCell>
                        <Badge variant={member.availability ? 'success' : 'secondary'}>
                          {member.availability ? 'Available' : 'Unavailable'}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                No staff data available
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Schedule Overview</CardTitle>
            <CardDescription>Generated schedule for selected period</CardDescription>
          </CardHeader>
          <CardContent>
            {scheduleLoading ? (
              <div className="flex items-center justify-center py-8">Loading schedule...</div>
            ) : schedule && schedule.length > 0 ? (
              <div className="space-y-4">
                {schedule.slice(0, 5).map((s) => {
                  const staffMember = staff?.find((st) => st.id === s.staff_id)
                  return (
                    <div key={s.staff_id} className="rounded-lg border p-4">
                      <p className="font-medium">{staffMember?.name || s.staff_id}</p>
                      <p className="text-sm text-muted-foreground">
                        {s.shifts.length} shift(s) scheduled
                      </p>
                    </div>
                  )
                })}
                {schedule.length > 5 && (
                  <p className="text-sm text-muted-foreground">
                    +{schedule.length - 5} more staff members
                  </p>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8 text-muted-foreground">
                No schedule generated. Click "Generate Schedule" to create one.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {schedule && schedule.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Detailed Schedule</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {schedule.map((s) => {
                const staffMember = staff?.find((st) => st.id === s.staff_id)
                return (
                  <div key={s.staff_id} className="space-y-2">
                    <h3 className="font-semibold">{staffMember?.name || s.staff_id}</h3>
                    <div className="grid gap-2 md:grid-cols-3">
                      {s.shifts.map((shift, idx) => (
                        <div key={idx} className="rounded border p-3 text-sm">
                          <p className="font-medium">{formatDate(shift.date)}</p>
                          <p className="text-muted-foreground">
                            {shift.start_time} - {shift.end_time}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

