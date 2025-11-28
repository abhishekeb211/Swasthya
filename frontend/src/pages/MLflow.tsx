import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Database } from 'lucide-react'
import { MLFLOW_URL } from '@/lib/api/client'

export function MLflow() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">MLflow Dashboard</h1>
        <p className="text-muted-foreground">Model tracking and experiment management</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            MLflow UI
          </CardTitle>
          <CardDescription>
            Embedded MLflow dashboard for experiment tracking and model registry
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[calc(100vh-300px)] w-full rounded-lg border">
            <iframe
              src={MLFLOW_URL}
              className="h-full w-full"
              title="MLflow Dashboard"
              style={{ border: 'none' }}
            />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            If the MLflow UI doesn't load, ensure the MLflow server is running on port 5000.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

