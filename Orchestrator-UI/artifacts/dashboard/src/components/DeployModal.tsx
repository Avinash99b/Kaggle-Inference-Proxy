import { useState, useEffect } from 'react';
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Loader2, AlertCircle } from 'lucide-react';

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
  const account = accountsData?.accounts?.find(a => a.account_id === accountId);
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

    createDeployment.mutate({
      data: {
        account_id: accountId,
        model_repo: repo,
        model_file: modelFile,
        model_name: modelName || modelFile.replace('.gguf', ''),
        estimated_quota_hours: parseFloat(hours)
      }
    }, {
      onSuccess: (data) => {
        toast({
          title: "Deployment started",
          description: data.notebook_url ? `Notebook: ${data.notebook_url}` : "Setting up worker...",
        });
        invalidateAll();
        onClose();
      },
      onError: (error) => {
        toast({
          variant: "destructive",
          title: "Failed to create deployment",
          description: error.error || "Unknown error occurred",
        });
      }
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[425px] border-border bg-card shadow-xl shadow-black/20" data-testid="modal-deploy">
        <DialogHeader>
          <DialogTitle className="text-foreground">Deploy Model</DialogTitle>
          <DialogDescription>
            Configure your LLM deployment for account <strong className="font-mono text-primary">{account?.username}</strong>
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-4 pt-4">
          <div className="space-y-2">
            <Label className="text-foreground">HuggingFace Repo ID</Label>
            <Input 
              placeholder="e.g. Qwen/Qwen3-8B-GGUF" 
              value={repo}
              onChange={e => setRepo(e.target.value)}
              className="bg-background font-mono text-sm border-border"
              data-testid="input-repo"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-foreground">Model File (.gguf)</Label>
            <Select 
              value={modelFile} 
              onValueChange={setModelFile} 
              disabled={!debouncedRepo || isFetchingFiles || ggufFiles.length === 0}
            >
              <SelectTrigger className="bg-background font-mono text-sm border-border" data-testid="select-modelfile">
                <SelectValue placeholder={
                  isFetchingFiles ? "Loading files…"
                  : filesError ? "Error loading files"
                  : (debouncedRepo && ggufFiles.length === 0) ? "No .gguf files found"
                  : "Select a model file"
                } />
              </SelectTrigger>
              <SelectContent>
                {ggufFiles.map(file => (
                  <SelectItem key={file} value={file} className="font-mono text-xs">{file}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            {filesError && (
              <p className="text-sm text-destructive flex items-center mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 mr-1 shrink-0" />
                {filesError.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label className="text-foreground">Deployment Name (Optional)</Label>
            <Input 
              placeholder="My awesome model" 
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              className="bg-background border-border"
              data-testid="input-modelname"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-foreground">Estimated Usage (Hours)</Label>
            <Input 
              type="number"
              step="0.1"
              min="0.1"
              value={hours}
              onChange={e => setHours(e.target.value)}
              className="bg-background font-mono border-border"
              data-testid="input-hours"
            />
            {isOverQuota && (
              <p className="text-sm text-destructive flex items-center mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 mr-1" />
                Exceeds remaining quota
              </p>
            )}
            {!isOverQuota && isWarnQuota && (
              <p className="text-sm text-amber-500 flex items-center mt-1 animate-in fade-in">
                <AlertCircle className="w-4 h-4 mr-1" />
                Uses over 80% of remaining quota
              </p>
            )}
          </div>

          <div className="flex justify-end space-x-2 pt-4">
            <Button type="button" variant="outline" onClick={onClose} data-testid="button-cancel">
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={isOverQuota || !modelFile || !repo || createDeployment.isPending}
              data-testid="button-submit-deploy"
            >
              {createDeployment.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Deploy
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
