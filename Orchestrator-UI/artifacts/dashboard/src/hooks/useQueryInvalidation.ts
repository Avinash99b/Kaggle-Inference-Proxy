import { useQueryClient } from '@tanstack/react-query';
import { getListAccountsQueryKey, getListDeploymentsQueryKey } from '@workspace/api-client-react';

export function useQueryInvalidation() {
  const queryClient = useQueryClient();

  const invalidateAccounts = () => {
    queryClient.invalidateQueries({ queryKey: getListAccountsQueryKey() });
  };

  const invalidateDeployments = () => {
    queryClient.invalidateQueries({ queryKey: getListDeploymentsQueryKey() });
  };

  const invalidateAll = () => {
    invalidateAccounts();
    invalidateDeployments();
  };

  return {
    invalidateAccounts,
    invalidateDeployments,
    invalidateAll,
  };
}
