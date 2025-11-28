import { useFLStatus, useFLHistory, useFLClients, useStartFLRound } from '@/hooks'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Brain, Play, Users } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export function FederatedLearning() {
  const { data: status1, isLoading: status1Loading } = useFLStatus(1)
  const { data: status2, isLoading: status2Loading } = useFLStatus(2)
  const { data: history1 } = useFLHistory(1)
  const { data: history2 } = useFLHistory(2)
  const { data: clients1 } = useFLClients(1)
  const { data: clients2 } = useFLClients(2)
  const startRound1 = useStartFLRound(1)
  const startRound2 = useStartFLRound(2)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Federated Learning Dashboard</h1>
        <p className="text-muted-foreground">Monitor federated learning rounds across servers</p>
      </div>

      {/* Server Status Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                FL Server 1
              </CardTitle>
              <Button
                onClick={() => startRound1.mutate()}
                disabled={startRound1.isPending}
                size="sm"
              >
                <Play className="mr-2 h-4 w-4" />
                Start Round
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {status1Loading ? (
              <div className="flex items-center justify-center py-4">Loading...</div>
            ) : status1 ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <Badge
                    variant={status1.status === 'active' ? 'success' : 'secondary'}
                  >
                    {status1.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Round ID</span>
                  <span className="text-sm font-medium">{status1.round_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Participants</span>
                  <span className="text-sm font-medium">{status1.participants}</span>
                </div>
                {status1.metrics && (
                  <div className="space-y-2 rounded-lg border p-3">
                    <p className="text-sm font-medium">Metrics</p>
                    {status1.metrics.accuracy && (
                      <p className="text-sm text-muted-foreground">
                        Accuracy: {(status1.metrics.accuracy * 100).toFixed(2)}%
                      </p>
                    )}
                    {status1.metrics.loss && (
                      <p className="text-sm text-muted-foreground">
                        Loss: {status1.metrics.loss.toFixed(4)}
                      </p>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center py-4 text-muted-foreground">
                No status available
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                FL Server 2
              </CardTitle>
              <Button
                onClick={() => startRound2.mutate()}
                disabled={startRound2.isPending}
                size="sm"
              >
                <Play className="mr-2 h-4 w-4" />
                Start Round
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {status2Loading ? (
              <div className="flex items-center justify-center py-4">Loading...</div>
            ) : status2 ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <Badge
                    variant={status2.status === 'active' ? 'success' : 'secondary'}
                  >
                    {status2.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Round ID</span>
                  <span className="text-sm font-medium">{status2.round_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Participants</span>
                  <span className="text-sm font-medium">{status2.participants}</span>
                </div>
                {status2.metrics && (
                  <div className="space-y-2 rounded-lg border p-3">
                    <p className="text-sm font-medium">Metrics</p>
                    {status2.metrics.accuracy && (
                      <p className="text-sm text-muted-foreground">
                        Accuracy: {(status2.metrics.accuracy * 100).toFixed(2)}%
                      </p>
                    )}
                    {status2.metrics.loss && (
                      <p className="text-sm text-muted-foreground">
                        Loss: {status2.metrics.loss.toFixed(4)}
                      </p>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center py-4 text-muted-foreground">
                No status available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Clients */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Server 1 Clients
            </CardTitle>
          </CardHeader>
          <CardContent>
            {clients1 && clients1.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Client ID</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {clients1.map((client) => (
                    <TableRow key={client.client_id}>
                      <TableCell>{client.client_id}</TableCell>
                      <TableCell>
                        <Badge variant={client.status === 'active' ? 'success' : 'secondary'}>
                          {client.status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="flex items-center justify-center py-4 text-muted-foreground">
                No clients connected
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Server 2 Clients
            </CardTitle>
          </CardHeader>
          <CardContent>
            {clients2 && clients2.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Client ID</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {clients2.map((client) => (
                    <TableRow key={client.client_id}>
                      <TableCell>{client.client_id}</TableCell>
                      <TableCell>
                        <Badge variant={client.status === 'active' ? 'success' : 'secondary'}>
                          {client.status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <div className="flex items-center justify-center py-4 text-muted-foreground">
                No clients connected
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* History Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {history1 && history1.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Server 1 Training History</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={history1}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round_id" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {history1[0]?.metrics?.accuracy && (
                    <Line
                      type="monotone"
                      dataKey="metrics.accuracy"
                      stroke="hsl(var(--primary))"
                      name="Accuracy"
                    />
                  )}
                  {history1[0]?.metrics?.loss && (
                    <Line
                      type="monotone"
                      dataKey="metrics.loss"
                      stroke="#ef4444"
                      name="Loss"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {history2 && history2.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Server 2 Training History</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={history2}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round_id" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {history2[0]?.metrics?.accuracy && (
                    <Line
                      type="monotone"
                      dataKey="metrics.accuracy"
                      stroke="hsl(var(--primary))"
                      name="Accuracy"
                    />
                  )}
                  {history2[0]?.metrics?.loss && (
                    <Line
                      type="monotone"
                      dataKey="metrics.loss"
                      stroke="#ef4444"
                      name="Loss"
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

