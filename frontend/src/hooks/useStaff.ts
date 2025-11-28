import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mockStaff } from '@/lib/api/mock'
import toast from 'react-hot-toast'

export function useStaff() {
  return useQuery({
    queryKey: ['staff'],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 400))
      return mockStaff
    },
  })
}

export function useGenerateSchedule() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: { start_date: string; end_date: string }) =>
      staffEndpoints.generateSchedule(data).then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['schedule'] })
      toast.success('Schedule generated successfully')
    },
    onError: () => {
      toast.error('Failed to generate schedule')
    },
  })
}

export function useSchedule(startDate: string, endDate: string) {
  return useQuery({
    queryKey: ['schedule', startDate, endDate],
    queryFn: () => staffEndpoints.getSchedule(startDate, endDate).then((res) => res.data),
    enabled: !!startDate && !!endDate,
  })
}

