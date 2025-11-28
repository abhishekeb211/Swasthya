import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mockForecast } from '@/lib/api/mock'
import toast from 'react-hot-toast'

export function useForecast(days: number = 7) {
  return useQuery({
    queryKey: ['forecast', days],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500))
      // Generate forecast for requested days
      return Array.from({ length: days }, (_, i) => {
        const date = new Date()
        date.setDate(date.getDate() + i)
        const base = 50 + Math.random() * 30 + Math.sin(i * 0.5) * 10
        return {
          date: date.toISOString().split('T')[0],
          predicted: Math.round(base),
          upper_bound: Math.round(base * 1.2),
          lower_bound: Math.round(base * 0.8),
        }
      })
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useTrainForecast() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      return forecastEndpoints.train(formData)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['forecast'] })
      toast.success('Training started successfully')
    },
    onError: () => {
      toast.error('Failed to start training')
    },
  })
}

export function useRunForecast() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => orchestratorEndpoints.runForecast(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['forecast'] })
      toast.success('Forecast workflow triggered')
    },
    onError: () => {
      toast.error('Failed to trigger forecast')
    },
  })
}

