import { useQuery } from '@tanstack/react-query';

interface HfTreeEntry {
  type: 'file' | 'directory';
  path: string;
  size: number;
  oid: string;
}

async function fetchGgufFiles(repo: string): Promise<string[]> {
  // Encode each path segment individually so the literal "/" is preserved —
  // HF rejects %2F (url-encoded slash) in the repo path.
  const encodedRepo = repo.split('/').map(encodeURIComponent).join('/');
  const url = `https://huggingface.co/api/models/${encodedRepo}/tree/main`;
  const res = await fetch(url);

  if (res.status === 404) {
    throw new Error(`Repository "${repo}" not found on Hugging Face.`);
  }
  if (!res.ok) {
    throw new Error(`Hugging Face API error: ${res.status} ${res.statusText}`);
  }

  const entries: HfTreeEntry[] = await res.json();
  return entries
    .filter((e) => e.type === 'file' && e.path.endsWith('.gguf'))
    .map((e) => e.path);
}

export function useHuggingFaceGgufFiles(repo: string) {
  return useQuery<string[], Error>({
    queryKey: ['hf-gguf-files', repo],
    queryFn: () => fetchGgufFiles(repo),
    enabled: repo.length >= 3,
    staleTime: 5 * 60 * 1000, // cache for 5 min; HF tree rarely changes mid-session
    retry: 1,
  });
}
