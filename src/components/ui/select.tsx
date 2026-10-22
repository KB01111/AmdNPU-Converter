import type { SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface SelectOption { value: string; label: string; disabled?: boolean }
interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "onChange"> {
  onValueChange: (value: string) => void; options: SelectOption[];
}
export function Select({ onValueChange, options, className, ...props }: SelectProps) {
  return <select {...props} onChange={event => onValueChange(event.target.value)} className={cn("h-9 w-full rounded-md border border-input bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50", className)}>
    {options.map(option => <option key={option.value} value={option.value} disabled={option.disabled}>{option.label}</option>)}
  </select>;
}
