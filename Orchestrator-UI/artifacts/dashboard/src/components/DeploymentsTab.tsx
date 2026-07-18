import { useState, useMemo } from 'react';
import { useListDeployments, useStopDeployment } from '@workspace/api-client-react';
import { useQueryInvalidation } from '@/hooks/useQueryInvalidation';
import { StatusBadge, ALL_STATUSES } from './StatusBadge';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { formatDistanceToNow } from 'date-fns';
import { Loader2, Square, ExternalLink, RefreshCw, Box, Eye, EyeOff } from 'lucide-react';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { getListDeploymentsQueryKey } from '@workspace/api-client-react';
import { useQueryClient } from '@tanstack/react-query';

const FILTER_COLORS: Record<string, string> = {
  all:       'border-border text-foreground bg-muted/40 hover:bg-muted',
  running:   'border-green-500/30 text-green-400 bg-green-500/10 hover:bg-green-500/20',
  queued:    'border-blue-500/30 text-blue-400 bg-blue-500/10 hover:bg-blue-500/20',
  created:   'border-amber-500/30 text-amber-400 bg-amber-500/10 hover:bg-amber-500/20',
  completed: 'border-sky-500/30 text-sky-400 bg-sky-500/10 hover:bg-sky-500/20',
  stopped:   'border-slate-500/30 text-slate-400 bg-slate-500/10 hover:bg-slate-500/20',
  error:     'border-red-500/30 text-red-400 bg-red-500/10 hover:bg-red-500/20',
};

const ACTIVE_FILTER_COLORS: Record<string, string> = {
  all:       'border-border bg-muted text-foreground',
  running:   'border-green-500/50 bg-green-500/20 text-green-300',
  queued:    'border-blue-500/50 bg-blue-500/20 text-blue-300',
  created:   'border-amber-500/50 bg-amber-500/20 text-amber-300',
  completed: 'border-sky-500/50 bg-sky-500/20 text-sky-300',
  stopped:   'border-slate-500/50 bg-slate-500/20 text-slate-300',
  error:     'border-red-500/50 bg-red-500/20 text-red-300',
};

type FilterKey = 'all' | (typeof ALL_STATUSES)[number];

const THREE_MINUTES_MS = 3 * 60 * 1000;

/** A deployment is visible by default if it's running OR was created within the last 3 minutes. */
function isDefaultVisible(dep: { notebook_status?: string | null; status?: string | null; created_at: number }): boolean {
  const isRunning = (dep.notebook_status ?? dep.status) === 'running';
  const isRecent = dep.created_at * 1000 > Date.now() - THREE_MINUTES_MS;
  return isRunning || isRecent;
}

export function DeploymentsTab() {
  const { data, isLoading, isFetching } = useListDeployments({ query: { refetchInterval: 10000 } });
  const stopDeployment = useStopDeployment();
  const { invalidateAll } = useQueryInvalidation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [activeFilter, setActiveFilter] = useState<FilterKey>('all');
  const [showAll, setShowAll] = useState(false);

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: getListDeploymentsQueryKey() });
  };

  const handleStop = (id: string) => {
    stopDeployment.mutate({ deploymentId: id }, {
      onSuccess: () => {
        toast({ title: 'Deployment stopped' });
        invalidateAll();
      },
      onError: (error) => {
        toast({
          variant: 'destructive',
          title: 'Failed to stop deployment',
          description: error.error || 'Unknown error occurred',
        });
      },
    });
  };

  const allDeployments = useMemo(
    () => data?.deployments
      ? [...data.deployments].sort((a, b) => b.created_at - a.created_at)
      : [],
    [data],
  );

  // Count per notebook_status for the filter chips
  const counts = useMemo(() => {
    const map: Record<string, number> = {};
    for (const dep of allDeployments) {
      const s = dep.notebook_status ?? dep.status ?? 'unknown';
      map[s] = (map[s] ?? 0) + 1;
    }
    return map;
  }, [allDeployments]);

  // Which filter keys actually have deployments (plus 'all' always shown)
  const visibleFilters: FilterKey[] = useMemo(() => {
    const present = new Set(allDeployments.map(d => d.notebook_status ?? d.status ?? 'unknown'));
    return ['all', ...ALL_STATUSES.filter(s => present.has(s))] as FilterKey[];
  }, [allDeployments]);

  // Apply the status filter chip
  const filtered = useMemo(
    () => activeFilter === 'all'
      ? allDeployments
      : allDeployments.filter(d => (d.notebook_status ?? d.status) === activeFilter),
    [allDeployments, activeFilter],
  );

  // Apply the show-all toggle: when off, hide non-running deployments older than 3 min
  const visibleDeployments = useMemo(
    () => showAll ? filtered : filtered.filter(isDefaultVisible),
    [filtered, showAll],
  );

  const hiddenCount = filtered.length - visibleDeployments.length;

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-foreground">Active Deployments</h2>
          <p className="text-sm text-muted-foreground">Monitor and manage your running model instances.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={showAll ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setShowAll(v => !v)}
            className={cn(
              'gap-1.5 h-8 px-2.5 text-xs font-mono border',
              showAll
                ? 'border-border/80 text-foreground'
                : 'border-border/40 text-muted-foreground hover:text-foreground',
            )}
            data-testid="button-show-all"
          >
            {showAll ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            {showAll ? 'Showing all' : 'Show all'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isFetching}
            data-testid="button-refresh-deployments"
          >
            <RefreshCw className={cn('w-4 h-4 mr-2', isFetching && 'animate-spin')} />
            Refresh Now
          </Button>
        </div>
      </div>

      {/* Filter chips — only render once data has loaded */}
      {!isLoading && allDeployments.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {visibleFilters.map((key) => {
            const count = key === 'all' ? allDeployments.length : (counts[key] ?? 0);
            const isActive = activeFilter === key;
            return (
              <button
                key={key}
                onClick={() => setActiveFilter(key)}
                className={cn(
                  'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-all duration-150',
                  isActive ? ACTIVE_FILTER_COLORS[key] : FILTER_COLORS[key],
                )}
                data-testid={`filter-${key}`}
              >
                <span className="capitalize">{key}</span>
                <span className={cn(
                  'inline-flex items-center justify-center rounded-full w-4 h-4 text-[10px] font-semibold',
                  isActive ? 'bg-white/20' : 'bg-black/10',
                )}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Hidden count hint */}
      {!isLoading && !showAll && hiddenCount > 0 && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono px-1">
          <EyeOff className="w-3 h-3 shrink-0" />
          <span>
            {hiddenCount} older, non-running deployment{hiddenCount !== 1 ? 's' : ''} hidden —{' '}
            <button onClick={() => setShowAll(true)} className="text-primary hover:underline">
              show all
            </button>
          </span>
        </div>
      )}

      {/* Content */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="animate-pulse bg-card border-border/50">
              <CardHeader className="pb-2"><div className="h-5 bg-muted rounded w-2/3" /></CardHeader>
              <CardContent><div className="h-10 bg-muted rounded w-full" /></CardContent>
              <CardFooter><div className="h-8 bg-muted rounded w-1/3 ml-auto" /></CardFooter>
            </Card>
          ))}
        </div>
      ) : visibleDeployments.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center border rounded-lg bg-card/50 border-dashed border-border/60">
          <Box className="w-10 h-10 text-muted-foreground/30 mb-4" />
          {allDeployments.length === 0 ? (
            <>
              <h3 className="text-lg font-medium text-foreground">No active deployments</h3>
              <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                Deploy a model from the accounts tab to see it here.
              </p>
            </>
          ) : hiddenCount > 0 ? (
            <>
              <h3 className="text-lg font-medium text-foreground">All older deployments are hidden</h3>
              <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                {hiddenCount} deployment{hiddenCount !== 1 ? 's' : ''} older than 3 minutes
                {activeFilter !== 'all' ? ` with status "${activeFilter}"` : ''}{' '}
                {hiddenCount !== 1 ? 'are' : 'is'} hidden.
              </p>
              <button
                onClick={() => setShowAll(true)}
                className="mt-3 text-xs text-primary hover:underline font-mono"
              >
                Show all
              </button>
            </>
          ) : (
            <>
              <h3 className="text-lg font-medium text-foreground">No matching deployments</h3>
              <p className="text-sm text-muted-foreground mt-1">
                No deployments with status <span className="font-mono">"{activeFilter}"</span>.
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {visibleDeployments.map((dep) => {
            const notebookStatus: string = dep.notebook_status ?? dep.status ?? 'unknown';
            const isRunning = notebookStatus === 'running';
            const isPending = stopDeployment.isPending && stopDeployment.variables?.deploymentId === dep.deployment_id;

            return (
              <Card
                key={dep.deployment_id}
                className={cn(
                  'flex flex-col overflow-hidden border-border/50 bg-card/50 transition-all duration-300',
                  isRunning
                    ? 'hover:border-green-500/30 hover:bg-card shadow-green-500/5 shadow-md'
                    : 'hover:bg-card hover:border-border',
                )}
                data-testid={`card-deployment-${dep.deployment_id}`}
              >
                <CardHeader className="pb-3">
                  <div className="flex justify-between items-start mb-1">
                    <StatusBadge status={notebookStatus} />
                    <span
                      className="text-xs text-muted-foreground font-mono"
                      title={new Date(dep.created_at * 1000).toISOString()}
                    >
                      {formatDistanceToNow(new Date(dep.created_at * 1000), { addSuffix: true })}
                    </span>
                  </div>
                  <CardTitle className="text-base truncate text-foreground" title={dep.model_name}>
                    {dep.model_name}
                  </CardTitle>
                  <CardDescription className="text-xs font-mono truncate text-muted-foreground" title={dep.model_repo}>
                    {dep.model_repo}
                  </CardDescription>
                </CardHeader>

                <CardContent className="pb-4 flex-1">
                  <div className="bg-background/50 rounded-md p-3 space-y-2 border border-border/50">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-muted-foreground">Account</span>
                      <span className="font-mono text-foreground font-medium">{dep.account_id}</span>
                    </div>
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-muted-foreground">File</span>
                      <span
                        className="font-mono text-foreground font-medium truncate max-w-[160px]"
                        title={dep.model_file}
                      >
                        {dep.model_file}
                      </span>
                    </div>
                    {dep.quota_reserved_seconds != null && (
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-muted-foreground">Reserved</span>
                        <span className="font-mono text-foreground font-medium">
                          {(dep.quota_reserved_seconds / 3600).toFixed(1)}h
                        </span>
                      </div>
                    )}
                    {dep.error_message && (
                      <div className="mt-2 p-2 bg-destructive/10 text-destructive text-xs rounded-sm font-mono whitespace-pre-wrap break-all border border-destructive/20">
                        {dep.error_message}
                      </div>
                    )}
                  </div>
                </CardContent>

                <CardFooter className="pt-0 flex justify-between gap-2 border-t border-border/50 p-4 bg-muted/10">
                  {dep.notebook_url ? (
                    <Button
                      variant="secondary"
                      size="sm"
                      asChild
                      className="flex-1 border border-border/50 bg-background hover:bg-muted"
                      data-testid={`link-notebook-${dep.deployment_id}`}
                    >
                      <a href={dep.notebook_url} target="_blank" rel="noreferrer">
                        <ExternalLink className="w-3 h-3 mr-1.5" /> Notebook
                      </a>
                    </Button>
                  ) : (
                    <div className="flex-1" />
                  )}

                  {isRunning && (
                    <Button
                      variant="destructive"
                      size="sm"
                      className="flex-shrink-0"
                      onClick={() => handleStop(dep.deployment_id)}
                      disabled={isPending}
                      data-testid={`button-stop-${dep.deployment_id}`}
                    >
                      {isPending
                        ? <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                        : <Square className="w-3 h-3 mr-1.5 fill-current" />
                      }
                      Stop
                    </Button>
                  )}
                </CardFooter>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
