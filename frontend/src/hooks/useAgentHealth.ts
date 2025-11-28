import { useQuery } from '@tanstack/react-query'
import { mockAgentHealth } from '@/lib/api/mock'

export function useAgentHealth() {
  return useQuery({
    queryKey: ['agent-health'],
    queryFn: async () => {
      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 500))
      return mockAgentHealth
    },
    refetchInterval: 30000,
  })
}

