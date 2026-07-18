import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { PlayCircle, Square, Clock, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

const STATUS_CONFIG: Record<string, { label: string; icon: React.ReactNode; classes: string }> = {
  running: {
    label: 'Running',
    icon: <PlayCircle className="w-3 h-3 mr-1" />,
    classes: 'bg-green-500/10 text-green-400 border-green-500/25',
  },
  created: {
    label: 'Created',
    icon: <Clock className="w-3 h-3 mr-1" />,
    classes: 'bg-amber-500/10 text-amber-400 border-amber-500/25',
  },
  queued: {
    label: 'Queued',
    icon: <Loader2 className="w-3 h-3 mr-1 animate-spin" />,
    classes: 'bg-blue-500/10 text-blue-400 border-blue-500/25',
  },
  completed: {
    label: 'Completed',
    icon: <CheckCircle2 className="w-3 h-3 mr-1" />,
    classes: 'bg-sky-500/10 text-sky-400 border-sky-500/25',
  },
  stopped: {
    label: 'Stopped',
    icon: <Square className="w-3 h-3 mr-1 fill-current" />,
    classes: 'bg-slate-500/10 text-slate-400 border-slate-500/25',
  },
  error: {
    label: 'Error',
    icon: <AlertCircle className="w-3 h-3 mr-1" />,
    classes: 'bg-red-500/10 text-red-400 border-red-500/25',
  },
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const cfg = STATUS_CONFIG[status];
  if (!cfg) {
    return (
      <Badge variant="outline" className={cn('bg-slate-500/10 text-slate-400 border-slate-500/25 capitalize', className)}>
        {status}
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className={cn(cfg.classes, className)}>
      {cfg.icon}
      {cfg.label}
    </Badge>
  );
}

/** All known notebook_status values for filter chips */
export const ALL_STATUSES = ['running', 'queued', 'created', 'completed', 'stopped', 'error'] as const;
export type NotebookStatus = (typeof ALL_STATUSES)[number] | (string & {});
