import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mockERQueue } from '@/lib/api/mock'
import toast from 'react-hot-toast'

export interface TriageRequest {
  symptoms: string[]
  vitals: {
    temperature?: number
    blood_pressure?: string
    heart_rate?: number
    oxygen_saturation?: number
  }
  lab_readings?: {
    [key: string]: number
  }
}

export function useTriage() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (data: TriageRequest) => {
      await new Promise((resolve) => setTimeout(resolve, 800))
      // Calculate acuity based on symptoms and vitals
      let acuity = 3
      if (data.vitals.heart_rate && data.vitals.heart_rate > 120) acuity = 5
      else if (data.vitals.heart_rate && data.vitals.heart_rate > 100) acuity = 4
      if (data.vitals.oxygen_saturation && data.vitals.oxygen_saturation < 90) acuity = 5
      if (data.symptoms.some((s) => s.toLowerCase().includes('chest pain'))) acuity = 5
      
      return {
        acuity_level: acuity,
        explanation: `Patient assessed with ${data.symptoms.length} symptoms. Vitals indicate ${acuity >= 4 ? 'critical' : acuity === 3 ? 'moderate' : 'mild'} condition.`,
        recommended_action: acuity >= 4 ? 'Immediate treatment required' : acuity === 3 ? 'Priority treatment' : 'Standard care',
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['er-queue'] })
      toast.success('Triage assessment completed')
    },
    onError: () => {
      toast.error('Triage assessment failed')
    },
  })
}

export function useERQueue() {
  return useQuery({
    queryKey: ['er-queue'],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 300))
      return mockERQueue
    },
    refetchInterval: 30000,
  })
}

export function useNextPatient() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => erOrEndpoints.getNextPatient().then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['er-queue'] })
      toast.success('Next patient retrieved')
    },
    onError: () => {
      toast.error('Failed to get next patient')
    },
  })
}

export function useAddPatient() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: Partial<Patient>) => erOrEndpoints.addPatient(data).then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['er-queue'] })
      toast.success('Patient added to ER queue')
    },
    onError: () => {
      toast.error('Failed to add patient')
    },
  })
}

