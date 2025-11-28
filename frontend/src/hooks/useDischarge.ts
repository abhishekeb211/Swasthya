import { useQuery } from '@tanstack/react-query'
import { mockDischargeAnalysis } from '@/lib/api/mock'

export function useDischargeAnalysis() {
  return useQuery({
    queryKey: ['discharge-analysis'],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500))
      return mockDischargeAnalysis
    },
    refetchInterval: 60000,
  })
}

export function useSingleDischargeAnalysis(patientId: string) {
  return useQuery({
    queryKey: ['discharge-analysis', patientId],
    queryFn: () => dischargeEndpoints.analyzeSingle(patientId).then((res) => res.data),
    enabled: !!patientId,
  })
}

