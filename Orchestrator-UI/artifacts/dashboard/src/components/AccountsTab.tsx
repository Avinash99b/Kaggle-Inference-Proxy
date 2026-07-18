import { useState } from 'react';
import { useListAccounts } from '@workspace/api-client-react';
import { QuotaBar, formatSeconds } from './QuotaBar';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Rocket, RefreshCw, Circle } from 'lucide-react';
import { DeployModal } from './DeployModal';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { useQueryClient } from '@tanstack/react-query';
import { getListAccountsQueryKey } from '@workspace/api-client-react';
import { cn } from '@/lib/utils';

export function AccountsTab() {
  const { data, isLoading, isFetching } = useListAccounts({ query: { refetchInterval: 10000 } });
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: getListAccountsQueryKey() });
  };

  const sortedAccounts = data?.accounts 
    ? [...data.accounts].sort((a, b) => b.gpu_quota_remaining_seconds - a.gpu_quota_remaining_seconds)
    : [];

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">Kaggle Accounts</h2>
          <p className="text-sm text-muted-foreground">Manage your quota and deploy models across available accounts.</p>
        </div>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleRefresh} 
          disabled={isFetching}
          data-testid="button-refresh-accounts"
        >
          <RefreshCw className={cn("w-4 h-4 mr-2", isFetching && "animate-spin")} />
          Refresh Now
        </Button>
      </div>

      <div className="rounded-md border border-border/60 bg-card overflow-hidden shadow-sm shadow-black/5">
        <Table>
          <TableHeader className="bg-muted/30">
            <TableRow className="hover:bg-transparent border-border/60">
              <TableHead className="font-medium text-foreground">Account Name</TableHead>
              <TableHead className="font-medium text-foreground">Status</TableHead>
              <TableHead className="font-medium text-foreground">GPU Total</TableHead>
              <TableHead className="w-[200px] font-medium text-foreground">Quota Remaining</TableHead>
              <TableHead className="text-right font-medium text-foreground">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i} className="border-border/60">
                  <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-12" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-32" /></TableCell>
                  <TableCell className="text-right"><Skeleton className="h-8 w-24 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : sortedAccounts.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="h-32 text-center text-muted-foreground">
                  <div className="flex flex-col items-center justify-center">
                    <Circle className="w-8 h-8 text-muted-foreground/30 mb-2" />
                    <span>No accounts found.</span>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              sortedAccounts.map((account) => {
                const isOnline = (Date.now() / 1000) - account.last_quota_update < 300;
                return (
                  <TableRow key={account.account_id} className="group hover:bg-muted/20 border-border/60 transition-colors data-[testid]:row-account">
                    <TableCell className="font-medium font-mono text-sm text-foreground">
                      {account.username}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn("bg-background shadow-sm", isOnline ? "text-green-500 border-green-500/20" : "text-muted-foreground border-muted-foreground/20")}>
                        <Circle className={cn("w-2 h-2 mr-1.5 fill-current", isOnline ? "text-green-500" : "text-muted-foreground")} />
                        {isOnline ? 'Online' : 'Offline'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground font-mono text-sm">
                      {formatSeconds(account.gpu_quota_total_seconds)}
                    </TableCell>
                    <TableCell>
                      <QuotaBar 
                        total={account.gpu_quota_total_seconds} 
                        remaining={account.gpu_quota_remaining_seconds} 
                      />
                    </TableCell>
                    <TableCell className="text-right">
                      <Button 
                        size="sm" 
                        variant="secondary"
                        className="border-primary/20 hover:border-primary/40 hover:bg-primary hover:text-primary-foreground"
                        onClick={() => setSelectedAccountId(account.account_id)}
                        disabled={account.gpu_quota_remaining_seconds <= 0}
                        data-testid={`button-deploy-${account.account_id}`}
                      >
                        <Rocket className="w-4 h-4 mr-1.5" /> Deploy Model
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>

      <DeployModal 
        isOpen={!!selectedAccountId} 
        onClose={() => setSelectedAccountId(null)} 
        accountId={selectedAccountId || ''} 
      />
    </div>
  );
}
