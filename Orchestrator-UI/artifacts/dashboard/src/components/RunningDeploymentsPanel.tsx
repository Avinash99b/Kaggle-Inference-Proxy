import { useMemo, forwardRef } from 'react';
import { useListDeployments } from '@workspace/api-client-react';
import type { Deployment } from '@workspace/api-client-react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Activity, ExternalLink, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLiveClock } from '@/hooks/useLiveClock';
import {
  getSessionInfo,
  formatDuration,
  SESSION_LIMIT_SECONDS,
  URGENCY_COLORS,
  type SessionUrgency,
} from '@/lib/sessionLimit';

// ---------------------------------------------------------------------------
// Single deployment row
// ---------------------------------------------------------------------------

function DeploymentRow({ dep }: { dep: Deployment }) {
  const now = useLiveClock(1000);
  const sinceUnix = dep.started_at ?? dep.created_at;
  const { elapsedSeconds, remainingSeconds, progressFraction, urgency } = getSessionInfo(sinceUnix, now);
  const colors = URGENCY_COLORS[urgency];
  const pct = Math.round(progressFraction * 100);

  return (
    <div
      className={cn(
        'rounded-lg border p-3 transition-colors group',
        urgency === 'critical'
          ? 'border-red-500/25 bg-red-500/5 hover:bg-red-500/10'
          : urgency === 'warning'
          ? 'border-amber-500/20 bg-amber-500/5 hover:bg-amber-500/10'
          : 'border-border/40 bg-muted/20 hover:bg-muted/40',
      )}
    >
      {/* Row 1: model name + time remaining */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="min-w-0 flex-1">
          <p
            className="text-sm font-semibold text-foreground truncate leading-tight"
            title={dep.model_name}
          >
            {dep.model_name}
          </p>
          <p className="text-[11px] text-muted-foreground font-mono truncate mt-0.5" title={dep.account_id}>
            {dep.account_id}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1 shrink-0">
          {/* Time remaining — prominent */}
          <span className={cn('font-mono font-bold text-sm tabular-nums', colors.text)}>
            {formatDuration(remainingSeconds)}
          </span>
          <span className="text-[10px] text-muted-foreground/60 font-mono">remaining</span>
        </div>
      </div>

      {/* Row 2: progress bar */}
      <div className="mb-2">
        <div className="h-2 w-full bg-secondary/60 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-700',
              colors.bar,
              urgency === 'critical' && 'animate-pulse',
            )}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </div>
      </div>

      {/* Row 3: elapsed + urgency badge + notebook link */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Clock className="w-3 h-3 text-muted-foreground/50 shrink-0" />
          <span className="text-[11px] text-muted-foreground font-mono tabular-nums">
            {formatDuration(elapsedSeconds)} elapsed
          </span>
          <span className="text-[11px] text-muted-foreground/40 font-mono">
            / {formatDuration(SESSION_LIMIT_SECONDS)}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {urgency !== 'safe' && (
            <span
              className={cn(
                'inline-flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded-full border font-mono',
                colors.badge,
              )}
            >
              {urgency === 'critical' && (
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
              )}
              {urgency === 'critical' ? 'Expiring soon' : 'Running long'}
            </span>
          )}

          {dep.notebook_url && (
            <a
              href={dep.notebook_url}
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground/50 hover:text-foreground transition-colors opacity-0 group-hover:opacity-100"
              title="Open notebook"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trigger button
// ---------------------------------------------------------------------------

const TriggerButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { count: number; topUrgency: SessionUrgency | null }
>(({ count, topUrgency, ...props }, ref) => {
  return (
    <button
      ref={ref}
      {...props}
      className={cn(
        'relative flex items-center justify-center w-8 h-8 rounded-md transition-colors',
        count > 0 && topUrgency === 'critical'
          ? 'text-red-400 hover:text-red-300 hover:bg-red-500/10'
          : count > 0 && topUrgency === 'warning'
          ? 'text-amber-400 hover:text-amber-300 hover:bg-amber-500/10'
          : count > 0
          ? 'text-green-400 hover:text-green-300 hover:bg-green-500/10'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
        props.className,
      )}
      title={count > 0 ? `${count} running deployment${count !== 1 ? 's' : ''}` : 'No running deployments'}
      data-testid="button-running-deployments"
    >
      <Activity className={cn('w-4 h-4', count > 0 && topUrgency === 'critical' && 'animate-pulse')} />

      {count > 0 && (
        <span
          className={cn(
            'absolute -top-1 -right-1 flex items-center justify-center w-4 h-4 rounded-full text-[9px] font-bold text-white leading-none shadow-sm',
            topUrgency === 'critical'
              ? 'bg-red-500 shadow-red-500/40'
              : topUrgency === 'warning'
              ? 'bg-amber-500 shadow-amber-500/40'
              : 'bg-green-500 shadow-green-500/40',
            topUrgency === 'critical' && 'animate-pulse',
          )}
        >
          {count > 9 ? '9+' : count}
        </span>
      )}
    </button>
  );
});
TriggerButton.displayName = 'TriggerButton';

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export function RunningDeploymentsPanel() {
  const { data } = useListDeployments({ query: { refetchInterval: 10000 } });
  const now = useLiveClock(10_000); // coarse clock for urgency classification

  const running = useMemo(
    () =>
      (data?.deployments ?? [])
        .filter((d) => {
          const status = (d as { notebook_status?: string }).notebook_status ?? d.status;
          return status === 'running';
        })
        .sort((a, b) => (a.started_at ?? a.created_at) - (b.started_at ?? b.created_at)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data],
  );

  const count = running.length;

  // Determine the highest urgency across all running deployments
  const topUrgency: SessionUrgency | null = useMemo(() => {
    if (count === 0) return null;
    const urgencies = running.map((d) => getSessionInfo(d.started_at ?? d.created_at, now).urgency);
    if (urgencies.includes('critical')) return 'critical';
    if (urgencies.includes('warning')) return 'warning';
    return 'safe';
  }, [running, now]);

  const criticalCount = useMemo(
    () => running.filter((d) => getSessionInfo(d.started_at ?? d.created_at, now).urgency === 'critical').length,
    [running, now],
  );

  return (
    <Popover>
      <PopoverTrigger asChild>
        <TriggerButton count={count} topUrgency={topUrgency} />
      </PopoverTrigger>

      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-[22rem] p-0 bg-card border-border/60 shadow-xl shadow-black/30"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
          <div className="flex items-center gap-2">
            <Activity
              className={cn(
                'w-3.5 h-3.5',
                topUrgency === 'critical' ? 'text-red-400 animate-pulse'
                  : topUrgency === 'warning' ? 'text-amber-400'
                  : 'text-green-400',
              )}
            />
            <span className="text-sm font-semibold text-foreground">Running Now</span>
            {criticalCount > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded-full border bg-red-500/15 text-red-300 border-red-500/25 font-mono animate-pulse">
                {criticalCount} expiring
              </span>
            )}
          </div>
          {count > 0 && (
            <span className="text-xs text-muted-foreground font-mono">
              {count} deployment{count !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Body */}
        {count === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center px-4">
            <Activity className="w-7 h-7 text-muted-foreground/25 mb-2" />
            <p className="text-sm text-muted-foreground">No deployments running</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Start one from the Deployments tab.
            </p>
          </div>
        ) : (
          <ScrollArea className="max-h-[26rem]">
            <div className="p-3 space-y-2">
              {running.map((dep) => (
                <DeploymentRow key={dep.deployment_id} dep={dep} />
              ))}
            </div>
          </ScrollArea>
        )}

        {/* Footer */}
        {count > 0 && (
          <div className="px-4 py-2.5 border-t border-border/50 flex items-center justify-between">
            <p className="text-[11px] text-muted-foreground/50 font-mono">
              11h session cap · updates every 10s
            </p>
            <p className="text-[11px] text-muted-foreground/50 font-mono">
              oldest first
            </p>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
