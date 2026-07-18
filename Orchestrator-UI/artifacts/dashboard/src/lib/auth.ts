const SECRET_COOKIE = 'orch_secret';
const URL_COOKIE = 'orch_url';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readCookie(name: string): string | null {
  const match = document.cookie.match(
    new RegExp(`(?:^|; )${name}=([^;]*)`),
  );
  return match ? decodeURIComponent(match[1]) : null;
}

function writeCookie(name: string, value: string): void {
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; SameSite=Strict`;
}

function deleteCookie(name: string): void {
  document.cookie = `${name}=; path=/; SameSite=Strict; Max-Age=0`;
}

// ---------------------------------------------------------------------------
// Shared secret
// ---------------------------------------------------------------------------

/** Read the shared secret from the session cookie. Returns null if absent. */
export function getSecret(): string | null {
  return readCookie(SECRET_COOKIE);
}

/** Persist the secret as a session-only cookie (cleared when browser closes). */
export function setSecret(secret: string): void {
  writeCookie(SECRET_COOKIE, secret);
}

/** Remove the secret cookie immediately. */
export function clearSecret(): void {
  deleteCookie(SECRET_COOKIE);
}

// ---------------------------------------------------------------------------
// Orchestrator URL
// ---------------------------------------------------------------------------

/** Read the orchestrator base URL from the session cookie. Returns null if absent. */
export function getOrchestratorUrl(): string | null {
  return readCookie(URL_COOKIE);
}

/** Persist the orchestrator URL as a session-only cookie (cleared when browser closes). */
export function setOrchestratorUrl(url: string): void {
  writeCookie(URL_COOKIE, url);
}

/** Remove the orchestrator URL cookie immediately. */
export function clearOrchestratorUrl(): void {
  deleteCookie(URL_COOKIE);
}
