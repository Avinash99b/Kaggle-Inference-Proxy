/** Maximum continuous execution time per deployment, in seconds (11 hours). */
export const SESSION_LIMIT_SECONDS = 11 * 3600; // 39 600

export type SessionUrgency = 'safe' | 'warning' | 'critical';

export interface SessionInfo {
  elapsedSeconds: number;
  remainingSeconds: number;
  /** 0 – 1 fraction of the 11-hour cap consumed. */
  progressFraction: number;
  urgency: SessionUrgency;
}

/**
 * Compute session progress for a deployment that started at `sinceUnix`
 * (a Unix epoch in **seconds**).
 */
export function getSessionInfo(sinceUnix: number, nowMs: number = Date.now()): SessionInfo {
  const elapsedSeconds = Math.max(0, Math.floor((nowMs - sinceUnix * 1000) / 1000));
  const remainingSeconds = Math.max(0, SESSION_LIMIT_SECONDS - elapsedSeconds);
  const progressFraction = Math.min(1, elapsedSeconds / SESSION_LIMIT_SECONDS);

  let urgency: SessionUrgency = 'safe';
  if (elapsedSeconds >= 9.5 * 3600) urgency = 'critical'; // < 1.5 h left
  else if (elapsedSeconds >= 7 * 3600) urgency = 'warning'; // < 4 h left

  return { elapsedSeconds, remainingSeconds, progressFraction, urgency };
}

/** Format a duration in seconds to a human-readable string. */
export function formatDuration(seconds: number): string {
  if (seconds <= 0) return '0s';
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem > 0 ? `${h}h ${rem.toString().padStart(2, '0')}m` : `${h}h`;
}

/** Tailwind colour tokens keyed by urgency level. */
export const URGENCY_COLORS: Record<SessionUrgency, {
  text: string;
  bar: string;
  badge: string;
  border: string;
  glow: string;
}> = {
  safe:     { text: 'text-green-400',  bar: 'bg-green-500',  badge: 'bg-green-500/15 text-green-300 border-green-500/25',  border: 'border-green-500/20', glow: 'shadow-green-500/10' },
  warning:  { text: 'text-amber-400',  bar: 'bg-amber-500',  badge: 'bg-amber-500/15 text-amber-300 border-amber-500/25',  border: 'border-amber-500/30', glow: 'shadow-amber-500/10' },
  critical: { text: 'text-red-400',    bar: 'bg-red-500',    badge: 'bg-red-500/15 text-red-300 border-red-500/25',        border: 'border-red-500/30',   glow: 'shadow-red-500/15'  },
};
