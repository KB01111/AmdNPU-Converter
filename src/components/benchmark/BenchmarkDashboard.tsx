import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { runBenchmark } from "@/lib/tauri";
import type { BenchmarkResult } from "@/lib/types";

const ALL_DEVICES = ["cpu", "gpu", "npu"];

export default function BenchmarkDashboard() {
  const [modelPath, setModelPath] = React.useState("");
  const [devices, setDevices] = React.useState<string[]>(["cpu", "npu"]);
  const [results, setResults] = React.useState<BenchmarkResult[]>([]);

  const mutation = useMutation({
    mutationFn: () => runBenchmark({ modelPath: modelPath.trim(), devices, iterations: 50 }),
    onSuccess: (data) => setResults(data),
  });

  const toggleDevice = (d: string) => {
    setDevices((prev) => (prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]));
  };

  const maxThroughput = Math.max(0, ...results.map((r) => r.throughputTps));

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Benchmark</CardTitle>
          <CardDescription>
            Calls run_benchmark with a compiled model path and a device list.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-col gap-2 md:flex-row">
            <Input
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              placeholder="/path/to/compiled/model.onnx"
              className="flex-1"
            />
            <Button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending || !modelPath.trim() || devices.length === 0}
            >
              {mutation.isPending ? <Loader2 className="animate-spin" /> : <Play />}
              Run benchmark
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {ALL_DEVICES.map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => toggleDevice(d)}
                className={`rounded-md border px-3 py-1 text-sm ${
                  devices.includes(d)
                    ? "border-primary bg-primary text-white"
                    : "border-input bg-background"
                }`}
              >
                {d.toUpperCase()}
              </button>
            ))}
          </div>
          {mutation.isError && (
            <Alert variant="destructive">
              <AlertTitle>Benchmark failed</AlertTitle>
              <AlertDescription>
                {mutation.error instanceof Error ? mutation.error.message : String(mutation.error)}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>CPU vs GPU vs NPU comparison (pure div bars, no chart lib).</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="py-2 pr-4 font-medium">Device</th>
                    <th className="py-2 pr-4 font-medium">Latency (ms)</th>
                    <th className="py-2 pr-4 font-medium">Throughput (tok/s)</th>
                    <th className="py-2 pr-4 font-medium">Power (W)</th>
                    <th className="py-2 font-medium">Relative</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.device} className="border-b last:border-0">
                      <td className="py-2 pr-4">
                        <Badge variant="secondary">{r.device.toUpperCase()}</Badge>
                      </td>
                      <td className="py-2 pr-4 font-mono">{r.avgLatencyMs.toFixed(2)}</td>
                      <td className="py-2 pr-4 font-mono">{r.throughputTps.toFixed(1)}</td>
                      <td className="py-2 pr-4 font-mono">
                        {r.powerW != null ? r.powerW.toFixed(1) : "—"}
                      </td>
                      <td className="py-2">
                        <div className="h-3 w-48 overflow-hidden rounded bg-secondary">
                          <div
                            className="h-full rounded bg-primary"
                            style={{
                              width: `${
                                maxThroughput > 0 ? (r.throughputTps / maxThroughput) * 100 : 0
                              }%`,
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 space-y-2">
              {results.map((r) => (
                <div key={r.device} className="flex items-center gap-3 text-sm">
                  <span className="w-16 font-medium">{r.device.toUpperCase()}</span>
                  <div className="h-4 flex-1 overflow-hidden rounded bg-secondary">
                    <div
                      className="h-full rounded bg-primary transition-all"
                      style={{
                        width: `${maxThroughput > 0 ? (r.throughputTps / maxThroughput) * 100 : 0}%`,
                      }}
                    />
                  </div>
                  <span className="w-32 font-mono text-xs">
                    {r.throughputTps.toFixed(1)} tok/s
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
