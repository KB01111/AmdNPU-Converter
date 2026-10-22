import * as React from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getJobStatus, onJobProgress } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";
import type { JobStage, JobStatus } from "@/lib/types";

const STAGES: JobStage[] = [
  "inspection",
  "download",
  "export",
  "optimize",
  "quantize",
  "compile",
  "validate",
  "benchmark",
];

const STAGE_INDEX: Record<JobStage, number> = {
  inspection: 0,
  download: 1,
  export: 2,
  optimize: 3,
  quantize: 4,
  compile: 5,
  validate: 6,
  benchmark: 7,
  done: 8,
};

interface StageNodeData extends Record<string, unknown> {
  label: string;
  stage: JobStage;
  status: "pending" | "active" | "done" | "failed";
  progress: number;
  logs: string[];
}

type StageNode = Node<StageNodeData, "stage">;

function StageNodeView({ data }: { data: StageNodeData }) {
  const [expanded, setExpanded] = React.useState(false);
  const color =
    data.status === "done"
      ? "border-emerald-500"
      : data.status === "active"
        ? "border-primary"
        : data.status === "failed"
          ? "border-destructive"
          : "border-border";
  return (
    <div className={`w-56 rounded-lg border-2 bg-card p-3 shadow ${color}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold">{data.label}</span>
        <Badge
          variant={
            data.status === "done" ? "success" : data.status === "active" ? "default" : "secondary"
          }
        >
          {data.status}
        </Badge>
      </div>
      <div className="mt-2">
        <Progress value={data.progress} />
      </div>
      <Button variant="ghost" size="sm" className="mt-1 h-6 px-1 text-xs" onClick={() => setExpanded((e) => !e)}>
        {expanded ? "Hide logs" : "Show logs"}
      </Button>
      {expanded && (
        <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap rounded bg-muted p-2 text-[10px] leading-tight">
          {data.logs.length > 0 ? data.logs.slice(-30).join("\n") : "No logs yet."}
        </pre>
      )}
    </div>
  );
}

const nodeTypes = { stage: StageNodeView };

function buildNodes(job: JobStatus | null): StageNode[] {
  const activeIdx = job ? (STAGE_INDEX[job.stage] ?? 0) : -1;
  const failed = job?.status === "failed";
  return STAGES.map((stage, i) => {
    let status: StageNodeData["status"] = "pending";
    if (job) {
      if (failed && i === activeIdx) status = "failed";
      else if (i < activeIdx || job.status === "completed") status = "done";
      else if (i === activeIdx) status = "active";
    }
    const progress =
      job == null
        ? 0
        : i < activeIdx || job.status === "completed"
          ? 100
          : i === activeIdx
            ? Math.round(job.progress)
            : 0;
    return {
      id: stage,
      type: "stage",
      position: { x: (i % 4) * 240, y: Math.floor(i / 4) * 190 },
      data: {
        label: stage.charAt(0).toUpperCase() + stage.slice(1),
        stage,
        status,
        progress,
        logs: job?.logs ?? [],
      },
    };
  });
}

function buildEdges(): Edge[] {
  const edges: Edge[] = [];
  for (let i = 0; i < STAGES.length - 1; i++) {
    edges.push({
      id: `${STAGES[i]}-${STAGES[i + 1]}`,
      source: STAGES[i],
      target: STAGES[i + 1],
      animated: true,
    });
  }
  return edges;
}

export default function ConversionPipeline() {
  const activeJobId = useAppStore((s) => s.activeJobId);
  const jobs = useAppStore((s) => s.jobs);
  const upsertJob = useAppStore((s) => s.upsertJob);

  const activeJob: JobStatus | null = React.useMemo(
    () => jobs.find((j) => j.id === activeJobId) ?? null,
    [jobs, activeJobId],
  );

  React.useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    (async () => {
      try {
        unlisten = await onJobProgress((evt) => {
          const existing = useAppStore.getState().jobs.find((j) => j.id === evt.jobId);
          if (existing) {
            upsertJob({
              ...existing,
              stage: evt.stage,
              progress: evt.progress,
              logs: [...existing.logs, evt.message],
              updatedAt: new Date().toISOString(),
            });
          } else {
            void getJobStatus(evt.jobId)
              .then((full) => upsertJob(full))
              .catch(() => undefined);
          }
        });
      } catch {
        if (!cancelled) {
          /* event bridge unavailable outside Tauri; polling below still works */
        }
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [upsertJob]);

  // Poll active job while running
  React.useEffect(() => {
    if (!activeJobId) return;
    const current = useAppStore.getState().jobs.find((j) => j.id === activeJobId);
    if (current && (current.status === "completed" || current.status === "failed" || current.status === "cancelled")) {
      return;
    }
    const timer = window.setInterval(() => {
      void getJobStatus(activeJobId)
        .then((s) => upsertJob(s))
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeJobId, upsertJob]);

  const nodes = React.useMemo(() => buildNodes(activeJob), [activeJob]);
  const edges = React.useMemo(() => buildEdges(), []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Conversion pipeline {activeJob ? `— ${activeJob.modelId}` : "(no active job)"}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!activeJob && (
          <p className="mb-3 text-sm text-muted-foreground">
            Start a conversion from the Import tab to watch live progress here. Listening on Tauri
            event <code className="font-mono">job://progress</code>.
          </p>
        )}
        {activeJob && (
          <div className="mb-3 flex items-center gap-3 text-sm">
            <Badge
              variant={
                activeJob.status === "completed"
                  ? "success"
                  : activeJob.status === "failed"
                    ? "destructive"
                    : "default"
              }
            >
              {activeJob.status}
            </Badge>
            <span className="text-muted-foreground">
              Stage: {activeJob.stage} — {Math.round(activeJob.progress)}%
            </span>
            <Progress value={activeJob.progress} className="max-w-xs" />
          </div>
        )}
        <div className="h-[420px] w-full rounded-lg border">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      </CardContent>
    </Card>
  );
}
