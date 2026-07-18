import { useState, useEffect } from 'react';
import { Terminal, Lock, Eye, EyeOff, AlertCircle, Globe } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

interface LoginPageProps {
  /** When true, show the "Invalid shared secret" banner. */
  authError?: boolean;
  onSuccess: () => void;
}

export function LoginPage({ authError, onSuccess }: LoginPageProps) {
  const { login } = useAuth();
  const [url, setUrl] = useState('');
  const [secret, setSecretValue] = useState('');
  const [showSecret, setShowSecret] = useState(false);
  const [error, setError] = useState(authError ? 'Invalid shared secret — please try again.' : '');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // If the parent updates authError after mount (e.g. 401 from orchestrator)
  useEffect(() => {
    if (authError) setError('Invalid shared secret — please try again.');
  }, [authError]);

  const clearError = () => { if (error) setError(''); };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedUrl = url.trim().replace(/\/+$/, '');
    const trimmedSecret = secret.trim();

    if (!trimmedUrl) {
      setError('Please enter the orchestrator URL.');
      return;
    }
    try {
      new URL(trimmedUrl);
    } catch {
      setError('Please enter a valid URL (e.g. https://your-orchestrator.example.com).');
      return;
    }
    if (!trimmedSecret) {
      setError('Please enter the shared secret.');
      return;
    }

    setError('');
    setIsSubmitting(true);
    try {
      login(trimmedSecret, trimmedUrl);
      onSuccess();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-[100dvh] flex flex-col items-center justify-center bg-background px-4">
      {/* Subtle grid background */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            'linear-gradient(hsl(var(--border)) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--border)) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      <div className="w-full max-w-sm relative z-10">
        {/* Logo / Brand */}
        <div className="flex flex-col items-center mb-8 gap-3">
          <div className="bg-primary/10 p-3 rounded-xl border border-primary/20 shadow-lg shadow-primary/5">
            <Terminal className="w-6 h-6 text-primary" />
          </div>
          <div className="text-center">
            <h1 className="font-mono font-bold text-lg tracking-tight uppercase text-foreground">
              Orchestrator
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">Secure access required</p>
          </div>
        </div>

        {/* Card */}
        <div className="bg-card border border-border/60 rounded-xl shadow-xl shadow-black/20 p-6">
          {/* Auth error banner */}
          {error && (
            <div className="mb-4 flex items-start gap-2.5 px-3 py-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm animate-in fade-in slide-in-from-top-1 duration-200">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Orchestrator URL */}
            <div className="space-y-1.5">
              <Label className="text-sm text-foreground flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-muted-foreground" />
                Orchestrator URL
              </Label>
              <Input
                type="url"
                placeholder="https://your-orchestrator.example.com"
                value={url}
                onChange={(e) => { setUrl(e.target.value); clearError(); }}
                className="bg-background border-border font-mono text-sm"
                autoFocus
                autoComplete="url"
                data-testid="input-url"
              />
            </div>

            {/* Shared secret */}
            <div className="space-y-1.5">
              <Label className="text-sm text-foreground flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-muted-foreground" />
                Shared Secret
              </Label>
              <div className="relative">
                <Input
                  type={showSecret ? 'text' : 'password'}
                  placeholder="Enter the shared secret…"
                  value={secret}
                  onChange={(e) => { setSecretValue(e.target.value); clearError(); }}
                  className="bg-background border-border font-mono text-sm pr-10"
                  autoComplete="current-password"
                  data-testid="input-secret"
                />
                <button
                  type="button"
                  onClick={() => setShowSecret((s) => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  tabIndex={-1}
                  aria-label={showSecret ? 'Hide secret' : 'Show secret'}
                >
                  {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={isSubmitting || !url.trim() || !secret.trim()}
              data-testid="button-connect"
            >
              {isSubmitting ? 'Connecting…' : 'Connect'}
            </Button>
          </form>
        </div>

        <p className="text-center text-xs text-muted-foreground/60 mt-5">
          URL and secret are stored in session cookies and cleared when you close the browser.
        </p>
      </div>
    </div>
  );
}
