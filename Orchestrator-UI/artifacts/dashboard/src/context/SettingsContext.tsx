import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import { DEFAULT_SESSION_LIMIT_SECONDS } from '@/lib/sessionLimit';

const STORAGE_KEY = 'orch_session_limit_hours';

function readStoredHours(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return DEFAULT_SESSION_LIMIT_SECONDS / 3600;
    const parsed = parseFloat(raw);
    if (!isFinite(parsed) || parsed <= 0) return DEFAULT_SESSION_LIMIT_SECONDS / 3600;
    return parsed;
  } catch {
    return DEFAULT_SESSION_LIMIT_SECONDS / 3600;
  }
}

interface SettingsContextValue {
  /** Configured session limit in seconds. */
  sessionLimitSeconds: number;
  /** Configured session limit in hours (convenience alias). */
  sessionLimitHours: number;
  setSessionLimitHours: (hours: number) => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [hours, setHoursState] = useState<number>(() => readStoredHours());

  const setSessionLimitHours = useCallback((h: number) => {
    const clamped = Math.max(0.5, Math.min(24, h));
    setHoursState(clamped);
    try {
      localStorage.setItem(STORAGE_KEY, String(clamped));
    } catch {
      // localStorage unavailable — still works in-memory for the session
    }
  }, []);

  return (
    <SettingsContext.Provider
      value={{
        sessionLimitSeconds: hours * 3600,
        sessionLimitHours: hours,
        setSessionLimitHours,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>');
  return ctx;
}
