import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { flEndpoints } from '@/lib/api/endpoints'
import toast from 'react-hot-toast'

export function useFLStatus(server: 1 | 2) {
  return useQuery({
    queryKey: ['fl-status', server],
    queryFn: () => flEndpoints.getStatus(server).then((res) => res.data),
    refetchInterval: 10000, // Refetch every 10 seconds
  })
}

export function useFLHistory(server: 1 | 2) {
  return useQuery({
    queryKey: ['fl-history', server],
    queryFn: () => flEndpoints.getHistory(server).then((res) => res.data),
  })
}

export function useFLClients(server: 1 | 2) {
  return useQuery({
    queryKey: ['fl-clients', server],
    queryFn: () => flEndpoints.getClients(server).then((res) => res.data),
    refetchInterval: 5000, // Refetch every 5 seconds
  })
}

export function useStartFLRound(server: 1 | 2) {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => flEndpoints.startRound(server).then((res) => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['fl-status', server] })
      queryClient.invalidateQueries({ queryKey: ['fl-history', server] })
      toast.success(`FL round started on server ${server}`)
    },
    onError: () => {
      toast.error(`Failed to start FL round on server ${server}`)
    },
  })
}

