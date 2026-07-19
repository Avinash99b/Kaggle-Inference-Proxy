import { useState, useEffect } from 'react';

/**
 * Returns the current timestamp in milliseconds, updated every `intervalMs`.
 * Multiple consumers with the same interval do NOT share a single timer —
 * each call sets up its own interval. For the accuracy needed here (session
 * progress bars) this is fine.
 */
export function useLiveClock(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
