import { cn } from '@/lib/utils';

export function formatSeconds(seconds: number): string {
  if (seconds < 0) return '0s';
  const hours = seconds / 3600;
  if (hours >= 1) {
    return `${hours.toFixed(1)}h`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}min`;
}

interface QuotaBarProps {
  total: number;
  remaining: number;
}

export function QuotaBar({ total, remaining }: QuotaBarProps) {
  const percentage = total > 0 ? (remaining / total) * 100 : 0;
  
  let colorClass = 'bg-red-500';
  if (percentage > 50) colorClass = 'bg-green-500';
  else if (percentage > 25) colorClass = 'bg-amber-500';

  return (
    <div className="flex flex-col gap-1.5 w-full max-w-[200px]" data-testid={`quota-bar-${remaining}`}>
      <div className="flex justify-between items-center text-xs">
        <span className="text-muted-foreground font-mono">{formatSeconds(remaining)} left</span>
        <span className="text-muted-foreground opacity-50 font-mono text-[10px]">{Math.round(percentage)}%</span>
      </div>
      <div className="h-1.5 w-full bg-secondary overflow-hidden rounded-full">
        <div 
          className={cn("h-full transition-all duration-500", colorClass)} 
          style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }} 
        />
      </div>
    </div>
  );
}
