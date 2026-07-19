import { useState, useEffect, useMemo } from 'react';
import { useCreateDeployment, useListAccounts } from '@workspace/api-client-react';
import { useHuggingFaceGgufFiles } from '@/hooks/useHuggingFaceGgufFiles';
import { useQueryInvalidation } from '@/hooks/useQueryInvalidation';
import { useToast } from '@/hooks/use-toast';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import {
  Loader2,
  AlertCircle,
  ChevronDown,
  Check,
  HardDrive,
  Star,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${bytes} B`;
}

/**
 * Extract the quantisation label from a GGUF filename.
 * Examples: "model-Q4_K_M.gguf" → "Q4_K_M", "Qwen3-8B-F16.gguf" → "F16"
 */
function extractQuant(filename: string | undefined): string | null {
  if (!filename) return null;
  // Strip directory prefix, look for quant tag just before .gguf
  const base = filename.split('/').pop() ?? filename;
  const m = base.match(/[-._]((?:IQ|Q|F|BF)\d+[\w]*)\.gguf$/i);
  return m ? m[1].toUpperCase() : null;
}

const RECOMMENDED_QUANTS = new Set(['Q4_K_M', 'Q5_K_M', 'Q6_K']);

// ---------------------------------------------------------------------------
// Model file combobox
// ---------------------------------------------------------------------------

interface ModelFilePickerProps {
  files: { path: string; size: number }[];
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
  isLoading: boolean;
  placeholder: string;
}

function ModelFilePicker({
  files,
  value,
  onChange,
  disabled,
  isLoading,
  placeholder,
}: ModelFilePickerProps) {
  const [open, setOpen] = useState(false);

  const selected = useMemo(() => files.find((f) => f.path === value) ?? null, [files, value]);

  const handleSelect = (path: string) => {
    onChange(path === value ? '' : path);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={disabled ? undefined : setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          data-testid="select-modelfile"
          className={cn(
            'flex w-full items-center justify-between rounded-md border bg-background px-3 py-2',
            'text-sm ring-offset-background transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'border-border hover:border-border/80',
            open && 'ring-2 ring-ring ring-offset-2',
          )}
          onClick={() => !disabled && setOpen((v) => !v)}
        >
          <span
            className={cn(
              'truncate font-mono text-xs text-left',
              !selected && 'text-muted-foreground font-sans',
            )}
          >
            {isLoading ? (
              <span className="flex items-center gap-2 text-muted-foreground font-sans">
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading files…
              </span>
            ) : selected ? (
              selected.path.split('/').pop()
            ) : (
              placeholder
            )}
          </span>

          <div className="flex items-center gap-1.5 shrink-0 ml-2">
            {selected && (
              <span className="text-[10px] text-muted-foreground/70 font-mono">
                {formatSize(selected.size)}
              </span>
            )}
            <ChevronDown className={cn('w-4 h-4 text-muted-foreground/60 transition-transform', open && 'rotate-180')} />
          </div>
        </button>
      </PopoverTrigger>

      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0 border-border shadow-xl shadow-black/25"
        align="start"
        sideOffset={4}
      >
        <Command>
          <div className="border-b border-border/60">
            <CommandInput
              placeholder="Search files…"
              className="h-9 text-xs font-mono"
            />
          </div>

          <CommandList className="max-h-[280px] overflow-y-auto overscroll-contain">
            <CommandEmpty className="py-6 text-center text-sm text-muted-foreground">
              No matching files.
            </CommandEmpty>

            <CommandGroup>
              {files.map((file) => {
                const quant = extractQuant(file.path);
                const isRecommended = quant ? RECOMMENDED_QUANTS.has(quant) : false;
                const displayName = file.path.split('/').pop() ?? file.path;
                const isSelected = file.path === value;

                return (
                  <CommandItem
                    key={file.path}
                    value={file.path}
                    onSelect={handleSelect}
                    className={cn(
                      'flex items-center gap-2 px-3 py-2.5 cursor-pointer aria-selected:bg-accent',
                      isSelected && 'bg-primary/5',
                    )}
                  >
                    {/* Check mark */}
                    <Check
                      className={cn(
                        'w-3.5 h-3.5 shrink-0',
                        isSelected ? 'text-primary opacity-100' : 'opacity-0',
                      )}
                    />

                    {/* File info */}
                    <div className="flex flex-col min-w-0 flex-1">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span
                          className={cn(
                            'font-mono text-xs truncate',
                            isSelected ? 'text-primary font-medium' : 'text-foreground',
                          )}
                          title={displayName}
                        >
                          {displayName}
                        </span>

                        {isRecommended && (
                          <span className="inline-flex items-center gap-0.5 text-[9px] font-bold px-1 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 shrink-0">
                            <Star className="w-2 h-2 fill-current" />
                            Recommended
                          </span>
                        )}

                        {quant && !isRecommended && (
                          <span className="inline-flex items-center text-[9px] font-bold px-1 py-0.5 rounded bg-secondary text-muted-foreground shrink-0 font-mono">
                            {quant}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Size */}
                    <div className="flex items-center gap-1 shrink-0 text-[11px] text-muted-foreground/70 font-mono">
                      <HardDrive className="w-3 h-3 opacity-50" />
                      {formatSize(file.size)}
                    </div>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>

          {files.length > 0 && (
            <div className="border-t border-border/50 px-3 py-1.5 flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground/50 font-mono">
                {files.length} file{files.length !== 1 ? 's' : ''} · sorted by size
              </span>
              {selected && (
                <button
                  type="button"
                  className="text-[10px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                  onClick={() => { onChange(''); setOpen(false); }}
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </Command>
      </PopoverContent>
    </Popover>
  );
}

// ---------------------------------------------------------------------------
// DeployModal
// ---------------------------------------------------------------------------

interface DeployModalProps {
  isOpen: boolean;
  onClose: () => void;
  accountId: string;
}

export function DeployModal({ isOpen, onClose, accountId }: DeployModalProps) {
  const [repo, setRepo] = useState('');
  const [debouncedRepo, setDebouncedRepo] = useState('');
  const [modelFile, setModelFile] = useState('');
  const [modelName, setModelName] = useState('');
  const [hours, setHours] = useState('1');

  const { toast } = useToast();
  const { invalidateAll } = useQueryInvalidation();
  const createDeployment = useCreateDeployment();

  const { data: accountsData } = useListAccounts();
  const account = accountsData?.accounts?.find((a) => a.account_id === accountId);
  const remainingSeconds = account?.gpu_quota_remaining_seconds || 0;

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedRepo(repo.length >= 3 ? repo : '');
    }, 300);
    return () => clearTimeout(timer);
  }, [repo]);

  useEffect(() => {
    setModelFile('');
  }, [debouncedRepo]);

  const {
    data: ggufFiles = [],
    isFetching: isFetchingFiles,
    error: filesError,
  } = useHuggingFaceGgufFiles(debouncedRepo);

  const requestedSeconds = parseFloat(hours) * 3600;
  const isOverQuota = requestedSeconds > remainingSeconds;
  const isWarnQuota = requestedSeconds > remainingSeconds * 0.8;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isOverQuota || !modelFile || !repo) return;

    createDeployment.mutate(
      {
        data: {
          account_id: accountId,
          model_repo: repo,
          model_file: modelFile,
          model_name: modelName || modelFile.split('/').pop()?.replace('.gguf', '') || modelFile,
          estimated_quota_hours: parseFloat(hours),
        },
      },
      {
        onSuccess: (data) => {
          toast({
            title: 'Deployment started',
            description: data.notebook_url
              ? `Notebook: ${data.notebook_url}`
              : 'Setting up worker…',
          });
          invalidateAll();
          onClose();
        },
        onError: (error) => {
          toast({
            variant: 'destructive',
            title: 'Failed to create deployment',
            description: error.error || 'Unknown error occurred',
          });
        },
      },
    );
  };

  // Picker disabled state + placeholder
  const pickerDisabled = !debouncedRepo || isFetchingFiles;
  const pickerPlaceholder = filesError
    ? 'Error loading files'
    : debouncedRepo && !isFetchingFiles && ggufFiles.length === 0
    ? 'No .gguf files found'
    : 'Select a model file';

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="sm:max-w-[480px] border-border bg-card shadow-xl shadow-black/20"
        data-testid="modal-deploy"
      >
        <DialogHeader>
          <DialogTitle className="text-foreground">Deploy Model</DialogTitle>
          <DialogDescription>
            Configure your LLM deployment for account{' '}
            <strong className="font-mono text-primary">{account?.username}</strong>
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 pt-4">
          {/* HuggingFace Repo */}
          <div className="space-y-1.5">
            <Label className="text-foreground text-sm">HuggingFace Repo ID</Label>
            <Input
              placeholder="e.g. Qwen/Qwen3-8B-GGUF"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              className="bg-background font-mono text-sm border-border"
              data-testid="input-repo"
            />
          </div>

          {/* Model file picker */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-foreground text-sm">Model File (.gguf)</Label>
              {isFetchingFiles && (
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Loader2 className="w-3 h-3 animate-spin" /> Fetching…
                </span>
              )}
              {!isFetchingFiles && ggufFiles.length > 0 && (
                <span className="text-xs text-muted-foreground font-mono">
                  {ggufFiles.length} file{ggufFiles.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            <ModelFilePicker
              files={ggufFiles}
              value={modelFile}
              onChange={setModelFile}
              disabled={pickerDisabled || ggufFiles.length === 0}
              isLoading={isFetchingFiles}
              placeholder={pickerPlaceholder}
            />

            {filesError && (
              <p className="text-sm text-destructive flex items-center gap-1.5 mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {filesError.message}
              </p>
            )}
          </div>

          {/* Deployment Name */}
          <div className="space-y-1.5">
            <Label className="text-foreground text-sm">
              Deployment Name{' '}
              <span className="text-muted-foreground font-normal">(optional)</span>
            </Label>
            <Input
              placeholder={
                modelFile
                  ? modelFile.split('/').pop()?.replace('.gguf', '') || 'My awesome model'
                  : 'My awesome model'
              }
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="bg-background border-border"
              data-testid="input-modelname"
            />
          </div>

          {/* Hours */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-foreground text-sm">Estimated Usage (Hours)</Label>
              <span className="text-xs text-muted-foreground font-mono">
                {remainingSeconds > 0
                  ? `${(remainingSeconds / 3600).toFixed(1)}h remaining`
                  : 'No quota'}
              </span>
            </div>
            <Input
              type="number"
              step="0.1"
              min="0.1"
              value={hours}
              onChange={(e) => setHours(e.target.value)}
              className="bg-background font-mono border-border"
              data-testid="input-hours"
            />
            {isOverQuota && (
              <p className="text-sm text-destructive flex items-center gap-1.5 mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 shrink-0" />
                Exceeds remaining quota
              </p>
            )}
            {!isOverQuota && isWarnQuota && (
              <p className="text-sm text-amber-500 flex items-center gap-1.5 mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 shrink-0" />
                Uses over 80% of remaining quota
              </p>
            )}
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={onClose} data-testid="button-cancel">
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isOverQuota || !modelFile || !repo || createDeployment.isPending}
              data-testid="button-submit-deploy"
            >
              {createDeployment.isPending && (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              )}
              Deploy
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
