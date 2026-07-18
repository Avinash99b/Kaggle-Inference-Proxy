import { useEffect, useCallback, useState } from 'react';
import { QueryClient, QueryClientProvider, QueryCache, MutationCache } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Route, Switch, Router as WouterRouter, useLocation } from 'wouter';
import { Terminal, Wifi, WifiOff, Loader2, LogOut } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';

import { AccountsTab } from './components/AccountsTab';
import { DeploymentsTab } from './components/DeploymentsTab';
import { useWebSocket } from './hooks/useWebSocket';
import { LoginPage } from './pages/LoginPage';
import { AuthProvider, useAuth } from './context/AuthContext';

// ---------------------------------------------------------------------------
// QueryClient — dispatch auth:expired on 401 / 403 so AuthContext can logout
// ---------------------------------------------------------------------------
function isAuthError(error: unknown): boolean {
  const status = (error as { status?: number })?.status;
  return status === 401 || status === 403;
}

function dispatchAuthExpired() {
  window.dispatchEvent(new CustomEvent('auth:expired'));
}

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError(error) {
      if (isAuthError(error)) dispatchAuthExpired();
    },
  }),
  mutationCache: new MutationCache({
    onError(error) {
      if (isAuthError(error)) dispatchAuthExpired();
    },
  }),
  defaultOptions: {
    queries: { retry: (count, error) => !isAuthError(error) && count < 2 },
  },
});

// ---------------------------------------------------------------------------
// Dashboard — shown when authenticated
// ---------------------------------------------------------------------------
function Dashboard() {
  const { secret, orchestratorUrl, logout } = useAuth();
  const { status } = useWebSocket({
    secret,
    orchestratorUrl,
    onAuthError: () => logout('expired'),
  });

  return (
    <div className="min-h-[100dvh] flex flex-col bg-background text-foreground selection:bg-primary/30">
      <header className="border-b border-border/50 bg-card/80 backdrop-blur-sm sticky top-0 z-10 shadow-sm shadow-black/10">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="bg-primary/10 p-1.5 rounded-md border border-primary/20">
              <Terminal className="w-4 h-4 text-primary" />
            </div>
            <h1 className="font-mono font-bold text-sm tracking-tight uppercase text-foreground">
              Orchestrator
            </h1>
          </div>

          <div className="flex items-center gap-2">
            {/* WS status */}
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-background border border-border/60 text-xs font-mono shadow-inner shadow-black/5">
              {status === 'connected' && (
                <><Wifi className="w-3 h-3 text-green-500" /><span className="text-muted-foreground font-medium">Live</span></>
              )}
              {status === 'connecting' && (
                <><Loader2 className="w-3 h-3 text-amber-500 animate-spin" /><span className="text-muted-foreground font-medium">Connecting…</span></>
              )}
              {status === 'disconnected' && (
                <><WifiOff className="w-3 h-3 text-red-500" /><span className="text-muted-foreground font-medium">Offline</span></>
              )}
            </div>

            {/* Logout */}
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground gap-1.5 h-8 px-2.5"
              onClick={() => logout('manual')}
              data-testid="button-logout"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span className="text-xs font-mono">Logout</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
        <Tabs defaultValue="accounts" className="w-full">
          <div className="flex justify-center mb-8">
            <TabsList className="grid w-full max-w-md grid-cols-2 bg-card border border-border/60 shadow-sm p-1">
              <TabsTrigger
                value="accounts"
                className="data-[state=active]:bg-background data-[state=active]:shadow-sm text-sm"
                data-testid="tab-accounts"
              >
                Accounts
              </TabsTrigger>
              <TabsTrigger
                value="deployments"
                className="data-[state=active]:bg-background data-[state=active]:shadow-sm text-sm"
                data-testid="tab-deployments"
              >
                Deployments
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="accounts" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
            <AccountsTab />
          </TabsContent>
          <TabsContent value="deployments" className="mt-0 focus-visible:outline-none focus-visible:ring-0">
            <DeploymentsTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Route guards — redirect using useEffect to avoid updating during render
// ---------------------------------------------------------------------------
function LoginRoute({
  authError,
  onClearAuthError,
}: {
  authError: boolean;
  onClearAuthError: () => void;
}) {
  const { secret } = useAuth();
  const [, navigate] = useLocation();
  useEffect(() => {
    if (secret) navigate('/');
  }, [secret, navigate]);
  if (secret) return null;
  return (
    <LoginPage
      authError={authError}
      onSuccess={() => { onClearAuthError(); navigate('/'); }}
    />
  );
}

function DashboardRoute() {
  const { secret } = useAuth();
  const [, navigate] = useLocation();
  useEffect(() => {
    if (!secret) navigate('/login');
  }, [secret, navigate]);
  if (!secret) return null;
  return <Dashboard />;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------
function AppRoutes({
  authError,
  onClearAuthError,
}: {
  authError: boolean;
  onClearAuthError: () => void;
}) {
  return (
    <Switch>
      <Route path="/login">
        <LoginRoute authError={authError} onClearAuthError={onClearAuthError} />
      </Route>

      <Route path="/">
        <DashboardRoute />
      </Route>

      <Route>
        <div className="min-h-[100dvh] flex items-center justify-center bg-background text-foreground font-mono">
          404 — SYSTEM NOT FOUND
        </div>
      </Route>
    </Switch>
  );
}

// ---------------------------------------------------------------------------
// InnerApp — owns AuthProvider (needs useLocation from WouterRouter above)
// ---------------------------------------------------------------------------
function InnerApp() {
  const [, navigate] = useLocation();
  const [authError, setAuthError] = useState(false);

  const handleLogout = useCallback(
    (reason: 'manual' | 'expired') => {
      queryClient.clear();
      setAuthError(reason === 'expired');
      navigate('/login');
    },
    [navigate],
  );

  return (
    <AuthProvider onLogout={handleLogout}>
      <AppRoutes authError={authError} onClearAuthError={() => setAuthError(false)} />
    </AuthProvider>
  );
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------
function App() {
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <InnerApp />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
