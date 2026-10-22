import { useQuery } from "@tanstack/react-query";
import { Cpu, CircuitBoard, Zap, RefreshCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getHardware } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";
import type { PowerMode } from "@/lib/types";

export default function HardwareDashboard() {
  const hardwareInfo = useAppStore((s) => s.hardwareInfo);
  const setHardwareInfo = useAppStore((s) => s.setHardwareInfo);
  const powerMode = useAppStore((s) => s.powerMode);
  const setPowerMode = useAppStore((s) => s.setPowerMode);

  const query = useQuery({
    queryKey: ["hardware"],
    queryFn: async () => {
      const info = await getHardware();
      setHardwareInfo(info);
      if (info.powerMode) setPowerMode(info.powerMode);
      return info;
    },
    staleTime: 30_000,
    retry: 1,
  });

  const info = query.data ?? hardwareInfo;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Hardware dashboard</h2>
        <Button variant="outline" size="sm" onClick={() => void query.refetch()} disabled={query.isFetching}>
          {query.isFetching ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          Refresh
        </Button>
      </div>

      {query.isError && (
        <Alert variant="destructive">
          <AlertTitle>Failed to read hardware</AlertTitle>
          <AlertDescription>
            {query.error instanceof Error ? query.error.message : String(query.error)}
          </AlertDescription>
        </Alert>
      )}

      {!info && !query.isFetching && (
        <Alert>
          <AlertTitle>No hardware data</AlertTitle>
          <AlertDescription>
            Press Refresh to call get_hardware_info on the Rust backend.
          </AlertDescription>
        </Alert>
      )}

      {info && (
        <>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Cpu className="h-4 w-4" /> CPU
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm">
                <div className="font-medium">{info.cpu.name}</div>
                <div className="text-muted-foreground">
                  {info.cpu.cores} cores / {info.cpu.threads} threads
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <CircuitBoard className="h-4 w-4" /> GPU
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm">
                {info.gpus.length === 0 && (
                  <span className="text-muted-foreground">No GPU reported</span>
                )}
                {info.gpus.map((g) => (
                  <div key={g.name} className="mb-1">
                    <div className="font-medium">{g.name}</div>
                    {g.vendor && <div className="text-muted-foreground">{g.vendor}</div>}
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Zap className="h-4 w-4" /> NPU
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm">
                {info.npu ? (
                  <>
                    <div className="font-medium">{info.npu.name}</div>
                    <Badge variant={info.npu.available ? "success" : "destructive"} className="mt-1">
                      {info.npu.available ? "available" : "unavailable"}
                    </Badge>
                    {info.npu.tops != null && (
                      <div className="mt-1 text-muted-foreground">{info.npu.tops} TOPS</div>
                    )}
                  </>
                ) : (
                  <span className="text-muted-foreground">No NPU reported</span>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Execution providers</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-1.5">
                {info.executionProviders.length === 0 && (
                  <span className="text-sm text-muted-foreground">None</span>
                )}
                {info.executionProviders.map((ep) => (
                  <Badge key={ep} variant="secondary">
                    {ep}
                  </Badge>
                ))}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Drivers</CardTitle>
                <CardDescription>Driver stack reported by the backend.</CardDescription>
              </CardHeader>
              <CardContent>
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="py-1 pr-3 font-medium">Name</th>
                      <th className="py-1 pr-3 font-medium">Version</th>
                      <th className="py-1 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.drivers.map((d) => (
                      <tr key={d.name} className="border-b last:border-0">
                        <td className="py-1.5 pr-3">{d.name}</td>
                        <td className="py-1.5 pr-3 font-mono text-xs">{d.version}</td>
                        <td className="py-1.5">
                          <Badge variant="outline">{d.status}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Power mode</CardTitle>
                <CardDescription>Stored in app state and sent with conversions.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Select
                  value={powerMode}
                  onValueChange={(v) => setPowerMode(v as PowerMode)}
                  options={[
                    { value: "automatic", label: "Automatic" },
                    { value: "max-performance", label: "Max performance" },
                    { value: "battery-efficient", label: "Battery efficient" },
                  ]}
                />
                <p className="text-xs text-muted-foreground">
                  Backend reports: {info.powerMode}. Changing this updates the conversion request.
                </p>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
