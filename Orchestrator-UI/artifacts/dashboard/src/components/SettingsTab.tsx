import { useState, useEffect } from 'react';
import { useSettings } from '@/context/SettingsContext';
import { getSessionInfo, formatDuration, URGENCY_COLORS } from '@/lib/sessionLimit';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Clock, AlertTriangle, CheckCircle2, RotateCcw } from 'lucide-react';

const DEFAULT_HOURS = 11;
const MIN_HOURS = 0.5;
const MAX_HOURS = 24;
const STEP = 0.5;

/** Preview what a given elapsed fraction looks like at the configured limit. */
function UrgencyPreviewBar({ limitHours }: { limitHours: number }) {
  const limitSeconds = limitHours * 3600;
  // Show three sample points: 50%, 70%, 90% elapsed
  const samples = [
    { label: '50%', frac: 0.5 },
    { label: '70%', frac: 7 / 11 },       // warning threshold
    { label: '87%', frac: 9.5 / 11 },     // critical threshold
    { label: '100%', frac: 1.0 },
  ];

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
        Urgency thresholds at {limitHours}h
      </p>

      {/* Visual timeline */}
      <div className="relative h-4 bg-secondary rounded-full overflow-hidden">
        {/* safe → warning gradient */}
        <div className="absolute inset-y-0 left-0 bg-green-500/60 rounded-l-full" style={{ width: `${(7 / 11) * 100}%` }} />
        <div className="absolute inset-y-0 bg-amber-500/60" style={{ left: `${(7 / 11) * 100}%`, width: `${((9.5 - 7) / 11) * 100}%` }} />
        <div className="absolute inset-y-0 right-0 bg-red-500/60 rounded-r-full" style={{ left: `${(9.5 / 11) * 100}%` }} />
        {/* tick marks */}
        <div className="absolute inset-y-0 w-px bg-background/60" style={{ left: `${(7 / 11) * 100}%` }} />
        <div className="absolute inset-y-0 w-px bg-background/60" style={{ left: `${(9.5 / 11) * 100}%` }} />
      </div>

      {/* Labels */}
      <div className="grid grid-cols-3 gap-2">
        <div className="flex items-start gap-2 rounded-lg border border-green-500/20 bg-green-500/5 p-2.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-green-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-[11px] font-semibold text-green-400">Safe</p>
            <p className="text-[10px] text-muted-foreground font-mono">
              0 – {formatDuration(Math.round((7 / 11) * limitSeconds))} elapsed
            </p>
          </div>
        </div>
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 p-2.5">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-[11px] font-semibold text-amber-400">Warning</p>
            <p className="text-[10px] text-muted-foreground font-mono">
              {formatDuration(Math.round((7 / 11) * limitSeconds))} – {formatDuration(Math.round((9.5 / 11) * limitSeconds))}
            </p>
          </div>
        </div>
        <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/5 p-2.5">
          <Clock className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-[11px] font-semibold text-red-400">Critical</p>
            <p className="text-[10px] text-muted-foreground font-mono">
              &gt; {formatDuration(Math.round((9.5 / 11) * limitSeconds))} elapsed
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export function SettingsTab() {
  const { sessionLimitHours, setSessionLimitHours } = useSettings();

  // Local draft — only committed on Save
  const [draft, setDraft] = useState(sessionLimitHours);
  const [inputValue, setInputValue] = useState(String(sessionLimitHours));
  const [saved, setSaved] = useState(false);

  // Keep draft in sync if the context value changes externally (unlikely but safe)
  useEffect(() => {
    setDraft(sessionLimitHours);
    setInputValue(String(sessionLimitHours));
  }, [sessionLimitHours]);

  const isDirty = draft !== sessionLimitHours;

  const handleSlider = (val: number[]) => {
    const h = val[0];
    setDraft(h);
    setInputValue(String(h));
  };

  const handleInput = (raw: string) => {
    setInputValue(raw);
    const parsed = parseFloat(raw);
    if (isFinite(parsed) && parsed >= MIN_HOURS && parsed <= MAX_HOURS) {
      // Round to nearest 0.5
      setDraft(Math.round(parsed * 2) / 2);
    }
  };

  const handleSave = () => {
    setSessionLimitHours(draft);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setDraft(DEFAULT_HOURS);
    setInputValue(String(DEFAULT_HOURS));
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-2xl">
      <div>
        <h2 className="text-lg font-semibold tracking-tight text-foreground">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Configure dashboard behaviour. Settings are saved to your browser and persist across sessions.
        </p>
      </div>

      {/* Session limit card */}
      <div className="rounded-xl border border-border/60 bg-card shadow-sm shadow-black/5 overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-border/50 flex items-center gap-2.5">
          <div className="bg-primary/10 p-1.5 rounded-md border border-primary/20">
            <Clock className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Session Limit</h3>
            <p className="text-xs text-muted-foreground">
              Maximum continuous GPU runtime per deployment. Used to calculate time-remaining warnings.
            </p>
          </div>
        </div>

        <div className="p-5 space-y-6">
          {/* Slider + input row */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium text-foreground">Limit</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  min={MIN_HOURS}
                  max={MAX_HOURS}
                  step={STEP}
                  value={inputValue}
                  onChange={(e) => handleInput(e.target.value)}
                  className="w-20 h-8 text-sm font-mono text-right bg-background border-border"
                />
                <span className="text-sm text-muted-foreground font-mono">hours</span>
              </div>
            </div>

            <Slider
              min={MIN_HOURS}
              max={MAX_HOURS}
              step={STEP}
              value={[draft]}
              onValueChange={handleSlider}
              className="w-full"
            />

            <div className="flex justify-between text-[10px] text-muted-foreground/50 font-mono">
              <span>{MIN_HOURS}h</span>
              <span className={cn(
                'font-semibold text-xs transition-colors',
                draft !== DEFAULT_HOURS ? 'text-primary' : 'text-muted-foreground',
              )}>
                {draft}h selected
              </span>
              <span>{MAX_HOURS}h</span>
            </div>
          </div>

          {/* Urgency preview */}
          <UrgencyPreviewBar limitHours={draft} />

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground gap-1.5"
              onClick={handleReset}
              disabled={draft === DEFAULT_HOURS}
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset to default ({DEFAULT_HOURS}h)
            </Button>

            <div className="flex items-center gap-2">
              {saved && (
                <span className="text-xs text-green-400 font-mono animate-in fade-in">
                  Saved ✓
                </span>
              )}
              <Button
                size="sm"
                onClick={handleSave}
                disabled={!isDirty}
                className="min-w-[80px]"
              >
                {isDirty ? 'Save' : 'Saved'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
