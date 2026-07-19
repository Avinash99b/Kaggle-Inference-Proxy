/**
 * Default session cap (11 hours) used when the user hasn't configured one.
 * The actual limit used at runtime comes from SettingsContext.
 */
export const DEFAULT_SESSION_LIMIT_SECONDS = 11 * 3600; // 39 600 s

export type SessionUrgency = 'safe' | 'warning' | 'critical';

export interface SessionInfo {
  elapsedSeconds: number;
  remainingSeconds: number;
  /** 0 – 1 fraction of the configured limit consumed. */
  progressFraction: number;
  urgency: SessionUrgency;
}

/**
 * Compute session progress for a deployment that started at `sinceUnix`
 * (Unix epoch in **seconds**).
 *
 * Urgency thresholds are expressed as fractions of `limitSeconds` so they
 * scale automatically when the user changes their session limit:
 *   • warning  — elapsed > 63.6 % of limit  (≈ 7 h out of 11 h default)
 *   • critical — elapsed > 86.4 % of limit  (≈ 9.5 h out of 11 h default)
 */
export function getSessionInfo(
  sinceUnix: number,
  nowMs: number = Date.now(),
  limitSeconds: number = DEFAULT_SESSION_LIMIT_SECONDS,
): SessionInfo {
  const limit = Math.max(1, limitSeconds); // guard against 0
  const elapsedSeconds = Math.max(0, Math.floor((nowMs - sinceUnix * 1000) / 1000));
  const remainingSeconds = Math.max(0, limit - elapsedSeconds);
  const progressFraction = Math.min(1, elapsedSeconds / limit);

  let urgency: SessionUrgency = 'safe';
  if (progressFraction >= 7 / 11)  urgency = 'warning';
  if (progressFraction >= 9.5 / 11) urgency = 'critical';

  return { elapsedSeconds, remainingSeconds, progressFraction, urgency };
}

/** Format a duration in seconds to a compact human-readable string. */
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
}> = {
  safe:     { text: 'text-green-400',  bar: 'bg-green-500',  badge: 'bg-green-500/15 text-green-300 border-green-500/25',  border: 'border-green-500/20' },
  warning:  { text: 'text-amber-400',  bar: 'bg-amber-500',  badge: 'bg-amber-500/15 text-amber-300 border-amber-500/25',  border: 'border-amber-500/30' },
  critical: { text: 'text-red-400',    bar: 'bg-red-500',    badge: 'bg-red-500/15 text-red-300 border-red-500/25',        border: 'border-red-500/30'   },
};
