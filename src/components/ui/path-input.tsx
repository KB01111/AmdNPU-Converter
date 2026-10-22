import { useId } from "react";
import { Input } from "./input";
import { Button } from "./button";
import { choosePath, errorMessage, isTauriRuntime } from "@/lib/tauri";

export function PathInput({ label, value, onChange, directory = false, extension, onError, disabled = false, placeholder }: {
  label: string; value: string; onChange: (path: string) => void; directory?: boolean; extension?: string;
  onError: (message: string) => void; disabled?: boolean; placeholder?: string;
}) {
  const id = useId();
  async function browse() {
    try { const path = await choosePath(directory, extension); if (path) onChange(path); }
    catch (error) { onError(errorMessage(error)); }
  }
  return <div className="min-w-0 space-y-1.5">
    <label htmlFor={id} className="text-sm font-medium">{label}</label>
    <div className="flex gap-2"><Input id={id} value={value} disabled={disabled} onChange={e => onChange(e.target.value)} placeholder={placeholder} />
      <Button type="button" variant="outline" disabled={disabled || !isTauriRuntime()} aria-label={`Browse ${label.toLowerCase()}`} onClick={() => void browse()}>Browse</Button>
    </div>
  </div>;
}
