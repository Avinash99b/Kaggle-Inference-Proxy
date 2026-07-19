import { useQuery } from '@tanstack/react-query';

export interface GgufFile {
  path: string;
  /** File size in bytes as reported by the HF API. */
  size: number;
}

interface HfTreeEntry {
  type: 'file' | 'directory';
  path: string;
  size: number;
  oid: string;
}

async function fetchGgufFiles(repo: string): Promise<GgufFile[]> {
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
    .map((e) => ({ path: e.path, size: e.size }))
    .sort((a, b) => a.size - b.size); // smallest first
}

export function useHuggingFaceGgufFiles(repo: string) {
  return useQuery<GgufFile[], Error>({
    queryKey: ['hf-gguf-files-v2', repo],
    queryFn: () => fetchGgufFiles(repo),
    enabled: repo.length >= 3,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
}
