import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from 'react';
import { setBaseUrl, setAuthTokenGetter } from '@workspace/api-client-react';
import {
  getSecret, setSecret, clearSecret,
  getOrchestratorUrl, setOrchestratorUrl, clearOrchestratorUrl,
} from '@/lib/auth';

interface AuthContextValue {
  secret: string | null;
  orchestratorUrl: string | null;
  login: (secret: string, orchestratorUrl: string) => void;
  logout: (reason?: 'manual' | 'expired') => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
  /** Called after any logout so the router can redirect to /login. */
  onLogout: (reason: 'manual' | 'expired') => void;
}

export function AuthProvider({ children, onLogout }: AuthProviderProps) {
  const [secret, setSecretState] = useState<string | null>(() => {
    // Initialize synchronously so the auth token getter and base URL are ready
    // before the first render — prevents React Query from firing unauthenticated
    // or misdirected requests on page refresh before useEffect has run.
    const s = getSecret();
    const u = getOrchestratorUrl();
    if (s) setAuthTokenGetter(() => s);
    if (u) setBaseUrl(u);
    return s;
  });

  const [orchestratorUrl, setOrchestratorUrlState] = useState<string | null>(
    () => getOrchestratorUrl(),
  );

  // Keep a stable ref so the event listener always has the latest logout fn
  const logoutRef = useRef<((reason?: 'manual' | 'expired') => void) | null>(null);

  const logout = useCallback(
    (reason: 'manual' | 'expired' = 'manual') => {
      clearSecret();
      clearOrchestratorUrl();
      setSecretState(null);
      setOrchestratorUrlState(null);
      setAuthTokenGetter(null);
      setBaseUrl(null);
      onLogout(reason);
    },
    [onLogout],
  );

  logoutRef.current = logout;

  const login = useCallback((newSecret: string, newUrl: string) => {
    const trimmedUrl = newUrl.replace(/\/+$/, ''); // strip trailing slash
    setSecret(newSecret);
    setOrchestratorUrl(trimmedUrl);
    setSecretState(newSecret);
    setOrchestratorUrlState(trimmedUrl);
    setAuthTokenGetter(() => newSecret);
    setBaseUrl(trimmedUrl);
  }, []);

  // Keep the token getter and base URL in sync whenever they change
  useEffect(() => {
    setAuthTokenGetter(secret ? () => secret : null);
  }, [secret]);

  useEffect(() => {
    setBaseUrl(orchestratorUrl ?? null);
  }, [orchestratorUrl]);

  // Listen for auth-expiry events dispatched by the QueryCache error handler
  useEffect(() => {
    const handler = () => logoutRef.current?.('expired');
    window.addEventListener('auth:expired', handler);
    return () => window.removeEventListener('auth:expired', handler);
  }, []);

  return (
    <AuthContext.Provider value={{ secret, orchestratorUrl, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within <AuthProvider>');
  return ctx;
}
