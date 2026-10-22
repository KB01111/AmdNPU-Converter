import * as React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Cpu, Download, GitBranch, FlaskConical, Library } from "lucide-react";
import HuggingFaceImport from "@/features/huggingface/HuggingFaceImport";
import ConversionPipeline from "@/features/conversion/ConversionPipeline";
import ConversionSettings from "@/features/conversion/ConversionSettings";
import ModelInspector from "@/components/model-inspector/ModelInspector";
import HardwareDashboard from "@/components/hardware-dashboard/HardwareDashboard";
import BenchmarkDashboard from "@/components/benchmark/BenchmarkDashboard";
import ModelLibrary from "@/components/model-library/ModelLibrary";
import type { AppTab } from "@/lib/types";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

const TABS: { id: AppTab; label: string; icon: React.ReactNode }[] = [
  { id: "import", label: "Import", icon: <Download className="h-4 w-4" /> },
  { id: "pipeline", label: "Pipeline", icon: <GitBranch className="h-4 w-4" /> },
  { id: "hardware", label: "Hardware", icon: <Cpu className="h-4 w-4" /> },
  { id: "benchmark", label: "Benchmark", icon: <FlaskConical className="h-4 w-4" /> },
  { id: "library", label: "Library", icon: <Library className="h-4 w-4" /> },
];

export default function App() {
  const [tab, setTab] = React.useState<AppTab>("import");

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-primary font-bold text-white">
                N
              </span>
              <div>
                <h1 className="text-base font-semibold leading-tight">AMD NPU Model Converter</h1>
                <p className="text-xs text-muted-foreground">Tauri v2 desktop app</p>
              </div>
            </div>
            <nav className="flex gap-1 rounded-lg bg-muted p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    tab === t.id
                      ? "bg-background text-foreground shadow"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        <main className="mx-auto max-w-6xl space-y-4 px-4 py-6">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.18 }}
          >
            {tab === "import" && (
              <div className="space-y-4">
                <HuggingFaceImport />
                <ConversionSettings />
                <ModelInspector />
              </div>
            )}
            {tab === "pipeline" && (
              <div className="space-y-4">
                <ConversionSettings />
                <ConversionPipeline />
              </div>
            )}
            {tab === "hardware" && <HardwareDashboard />}
            {tab === "benchmark" && <BenchmarkDashboard />}
            {tab === "library" && <ModelLibrary />}
          </motion.div>
        </main>
      </div>
    </QueryClientProvider>
  );
}
