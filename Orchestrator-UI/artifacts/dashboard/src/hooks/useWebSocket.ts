import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getListAccountsQueryKey, getListDeploymentsQueryKey } from '@workspace/api-client-react';
import type { AccountsResponse, DeploymentsResponse } from '@workspace/api-client-react';

type WSStatus = 'connecting' | 'connected' | 'disconnected';

/** Close code sent by the orchestrator when the shared secret is rejected. */
const AUTH_CLOSE_CODE = 4401;

interface UseWebSocketOptions {
  secret: string | null;
  orchestratorUrl: string | null;
  onAuthError?: () => void;
}

/** Convert an http(s) URL to its ws(s) equivalent. */
function toWsUrl(httpUrl: string, secret: string): string {
  const wsBase = httpUrl.replace(/^http/, (m) => (m === 'https' ? 'wss' : 'ws'));
  const url = new URL('/ws', wsBase);
  url.searchParams.set('secret', secret);
  return url.toString();
}

export function useWebSocket({ secret, orchestratorUrl, onAuthError }: UseWebSocketOptions) {
  const [status, setStatus] = useState<WSStatus>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const onAuthErrorRef = useRef(onAuthError);
  onAuthErrorRef.current = onAuthError;
  const queryClient = useQueryClient();

  useEffect(() => {
    // Don't connect without both a secret and a URL
    if (!secret || !orchestratorUrl) {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setStatus('disconnected');
      return;
    }

    let timeoutId: number;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      setStatus('connecting');

      const wsUrl = toWsUrl(orchestratorUrl, secret);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        reconnectCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleEvent(data);
        } catch {
          console.error('Failed to parse WS message', event.data);
        }
      };

      ws.onclose = (evt) => {
        setStatus('disconnected');
        wsRef.current = null;

        // Auth rejection — don't reconnect, trigger logout
        if (evt.code === AUTH_CLOSE_CODE) {
          onAuthErrorRef.current?.();
          return;
        }

        if (stopped) return;
        const count = reconnectCountRef.current;
        const delays = [1000, 2000, 4000, 8000];
        const delay = count < delays.length ? delays[count] : 30_000;
        reconnectCountRef.current++;
        timeoutId = window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // onerror is always followed by onclose; let onclose handle reconnect
        ws.close();
      };
    };

    connect();

    return () => {
      stopped = true;
      clearTimeout(timeoutId);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  // Re-run whenever secret or orchestratorUrl changes (login/logout)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [secret, orchestratorUrl]);

  const handleEvent = (event: Record<string, unknown>) => {
    if (event['event'] === 'quota_update' && event['account_id']) {
      queryClient.setQueryData(
        getListAccountsQueryKey(),
        (old: AccountsResponse | undefined) => {
          if (!old) return old;
          return {
            ...old,
            accounts: old.accounts.map((acc) =>
              acc.account_id === event['account_id']
                ? { ...acc, gpu_quota_remaining_seconds: event['gpu_quota_remaining_seconds'] as number }
                : acc,
            ),
          };
        },
      );
    }

    if (event['event'] === 'deployment_status_changed' && event['deployment_id']) {
      queryClient.setQueryData(
        getListDeploymentsQueryKey(),
        (old: DeploymentsResponse | undefined) => {
          if (!old) return old;
          return {
            ...old,
            deployments: old.deployments.map((dep) =>
              dep.deployment_id === event['deployment_id']
                ? {
                    ...dep,
                    notebook_status: event['status'] as string,
                    notebook_url: (event['notebook_url'] as string | undefined) || dep.notebook_url,
                  }
                : dep,
            ),
          };
        },
      );
    }

    if (event['event'] === 'deployment_error' && event['deployment_id']) {
      queryClient.setQueryData(
        getListDeploymentsQueryKey(),
        (old: DeploymentsResponse | undefined) => {
          if (!old) return old;
          return {
            ...old,
            deployments: old.deployments.map((dep) =>
              dep.deployment_id === event['deployment_id']
                ? { ...dep, notebook_status: 'error', error_message: event['error'] as string }
                : dep,
            ),
          };
        },
      );
    }
  };

  return { status };
}
