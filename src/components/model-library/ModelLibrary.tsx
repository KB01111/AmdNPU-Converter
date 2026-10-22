import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { FolderOpen, Loader2, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cancelJob, getJobStatus, listJobs, listModels } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";
import { formatBytes } from "@/features/huggingface/HuggingFaceImport";

function statusVariant(s: string): "default" | "secondary" | "destructive" | "success" | "warning" | "outline" {
  switch (s) {
    case "completed":
      return "success";
    case "failed":
      return "destructive";
    case "cancelled":
      return "warning";
    case "running":
      return "default";
    default:
      return "secondary";
  }
}

export default function ModelLibrary() {
  const jobs = useAppStore((s) => s.jobs);
  const setJobs = useAppStore((s) => s.setJobs);
  const setActiveJobId = useAppStore((s) => s.setActiveJobId);
  const [cancelling, setCancelling] = React.useState<string | null>(null);

  const jobsQuery = useQuery({
    queryKey: ["jobs"],
    queryFn: async () => {
      const data = await listJobs();
      setJobs(data);
      return data;
    },
    staleTime: 10_000,
    retry: 1,
  });

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: () => listModels(),
    staleTime: 10_000,
    retry: 1,
  });

  const handleCancel = async (id: string) => {
    setCancelling(id);
    try {
      await cancelJob(id);
      const updated = await getJobStatus(id);
      setJobs(jobs.map((j) => (j.id === id ? updated : j)));
    } finally {
      setCancelling(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Model library & job history</h2>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              void jobsQuery.refetch();
              void modelsQuery.refetch();
            }}
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        </div>
      </div>

      {(jobsQuery.isError || modelsQuery.isError) && (
        <Alert variant="destructive">
          <AlertTitle>Failed to load library</AlertTitle>
          <AlertDescription>
            {[jobsQuery.error, modelsQuery.error]
              .filter(Boolean)
              .map((e) => (e instanceof Error ? e.message : String(e)))
              .join(" ")}
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Conversion jobs ({jobs.length})</CardTitle>
          <CardDescription>History from list_jobs. Click a job to inspect it in the pipeline.</CardDescription>
        </CardHeader>
        <CardContent>
          {jobsQuery.isFetching && jobs.length === 0 ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="animate-spin" /> Loading jobs…
            </div>
          ) : jobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No jobs yet. Start a conversion first.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="py-2 pr-3 font-medium">ID</th>
                    <th className="py-2 pr-3 font-medium">Model</th>
                    <th className="py-2 pr-3 font-medium">Status</th>
                    <th className="py-2 pr-3 font-medium">Stage</th>
                    <th className="py-2 pr-3 font-medium">Progress</th>
                    <th className="py-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.id} className="border-b last:border-0">
                      <td className="py-1.5 pr-3 font-mono text-xs">{j.id.slice(0, 8)}</td>
                      <td className="py-1.5 pr-3">{j.modelId}</td>
                      <td className="py-1.5 pr-3">
                        <Badge variant={statusVariant(j.status)}>{j.status}</Badge>
                      </td>
                      <td className="py-1.5 pr-3">{j.stage}</td>
                      <td className="py-1.5 pr-3 font-mono">{Math.round(j.progress)}%</td>
                      <td className="flex gap-1 py-1.5">
                        <Button variant="ghost" size="sm" onClick={() => setActiveJobId(j.id)}>
                          <FolderOpen className="h-4 w-4" /> Open
                        </Button>
                        {(j.status === "running" || j.status === "queued") && (
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={cancelling === j.id}
                            onClick={() => void handleCancel(j.id)}
                          >
                            <XCircle className="h-4 w-4" /> Cancel
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Converted models ({modelsQuery.data?.length ?? 0})</CardTitle>
          <CardDescription>Artifacts from list_models.</CardDescription>
        </CardHeader>
        <CardContent>
          {modelsQuery.isFetching && !modelsQuery.data ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="animate-spin" /> Loading models…
            </div>
          ) : !modelsQuery.data || modelsQuery.data.length === 0 ? (
            <p className="text-sm text-muted-foreground">No converted models yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="py-2 pr-3 font-medium">Model</th>
                    <th className="py-2 pr-3 font-medium">Format</th>
                    <th className="py-2 pr-3 font-medium">Size</th>
                    <th className="py-2 font-medium">Path</th>
                  </tr>
                </thead>
                <tbody>
                  {modelsQuery.data.map((m) => (
                    <tr key={m.id} className="border-b last:border-0">
                      <td className="py-1.5 pr-3 font-medium">{m.modelId}</td>
                      <td className="py-1.5 pr-3">
                        <Badge variant="outline">{m.format}</Badge>
                      </td>
                      <td className="py-1.5 pr-3">{formatBytes(m.sizeBytes)}</td>
                      <td className="py-1.5 font-mono text-xs text-muted-foreground">{m.path}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
