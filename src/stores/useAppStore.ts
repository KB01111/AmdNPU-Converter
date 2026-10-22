import { create } from "zustand";
import type {
  DevicePreference,
  HardwareInfo,
  JobStatus,
  ModelMetadata,
  PowerMode,
  Precision,
  TargetEp,
} from "@/lib/types";

interface ConversionSettingsState {
  precision: Precision;
  device: DevicePreference;
  powerMode: PowerMode;
  targetEp: TargetEp;
  setPrecision: (p: Precision) => void;
  setDevice: (d: DevicePreference) => void;
  setPowerMode: (m: PowerMode) => void;
  setTargetEp: (e: TargetEp) => void;
}

interface AppState extends ConversionSettingsState {
  huggingfaceUrl: string;
  modelMetadata: ModelMetadata | null;
  hardwareInfo: HardwareInfo | null;
  jobs: JobStatus[];
  activeJobId: string | null;
  setHuggingfaceUrl: (url: string) => void;
  setModelMetadata: (m: ModelMetadata | null) => void;
  setHardwareInfo: (h: HardwareInfo | null) => void;
  setJobs: (jobs: JobStatus[]) => void;
  upsertJob: (job: JobStatus) => void;
  setActiveJobId: (id: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  huggingfaceUrl: "",
  modelMetadata: null,
  hardwareInfo: null,
  jobs: [],
  activeJobId: null,

  precision: "auto",
  device: "auto",
  powerMode: "automatic",
  targetEp: "vitisai",

  setHuggingfaceUrl: (url) => set({ huggingfaceUrl: url }),
  setModelMetadata: (m) => set({ modelMetadata: m }),
  setHardwareInfo: (h) => set({ hardwareInfo: h }),
  setJobs: (jobs) => set({ jobs }),
  upsertJob: (job) =>
    set((s) => {
      const exists = s.jobs.some((j) => j.id === job.id);
      return {
        jobs: exists ? s.jobs.map((j) => (j.id === job.id ? job : j)) : [...s.jobs, job],
      };
    }),
  setActiveJobId: (id) => set({ activeJobId: id }),

  setPrecision: (precision) => set({ precision }),
  setDevice: (device) => set({ device }),
  setPowerMode: (powerMode) => set({ powerMode }),
  setTargetEp: (targetEp) => set({ targetEp }),
}));
