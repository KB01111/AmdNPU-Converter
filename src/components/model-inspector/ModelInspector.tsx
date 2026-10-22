import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { startConversion } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";
import { formatBytes } from "@/features/huggingface/HuggingFaceImport";

function recommendedPrecision(params: number): string {
  if (params >= 7_000_000_000) return "A8W8 / INT8 recommended for NPU memory";
  if (params >= 3_000_000_000) return "A16W8 recommended for NPU";
  return "BF16 or FP32 is fine for this size";
}

export default function ModelInspector() {
  const modelMetadata = useAppStore((s) => s.modelMetadata);
  const precision = useAppStore((s) => s.precision);
  const device = useAppStore((s) => s.device);
  const powerMode = useAppStore((s) => s.powerMode);
  const targetEp = useAppStore((s) => s.targetEp);
  const hardwareInfo = useAppStore((s) => s.hardwareInfo);
  const huggingfaceUrl = useAppStore((s) => s.huggingfaceUrl);
  const setActiveJobId = useAppStore((s) => s.setActiveJobId);
  const upsertJob = useAppStore((s) => s.upsertJob);
  const [error, setError] = React.useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => {
      if (!modelMetadata) throw new Error("No model inspected yet");
      return startConversion({
        modelId: modelMetadata.modelId,
        url: huggingfaceUrl.trim() || modelMetadata.url,
        precision,
        device,
        powerMode,
        targetEp,
        outputDir: null,
      });
    },
    onSuccess: async (jobId) => {
      setError(null);
      setActiveJobId(jobId);
      // Seed a queued job entry so the pipeline shows something immediately.
      const now = new Date().toISOString();
      upsertJob({
        id: jobId,
        modelId: modelMetadata?.modelId ?? "unknown",
        stage: "inspection",
        progress: 0,
        status: "queued",
        logs: [`Job ${jobId} queued for ${modelMetadata?.modelId ?? "model"}`],
        error: null,
        createdAt: now,
        updatedAt: now,
      });
    },
    onError: (e) => {
      setError(e instanceof Error ? e.message : String(e));
    },
  });

  if (!modelMetadata) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model inspector</CardTitle>
          <CardDescription>Inspect a Hugging Face model first.</CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Go to the Import tab and run Inspect. The CONVERT TO NPU action appears here.
        </CardContent>
      </Card>
    );
  }

  const npuAvailable = hardwareInfo?.npu?.available ?? null;
  const compatible = modelMetadata.format.toLowerCase().includes("safetensors");

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="h-5 w-5" /> {modelMetadata.modelId}
        </CardTitle>
        <CardDescription>
          {modelMetadata.arch} · {modelMetadata.paramsHuman} · {formatBytes(modelMetadata.totalSizeBytes)}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
          <div>
            <div className="text-muted-foreground">Arch</div>
            <div className="font-medium">{modelMetadata.arch}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Params</div>
            <div className="font-medium">{modelMetadata.params.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Format</div>
            <div className="font-medium">{modelMetadata.format}</div>
          </div>
          <div>
            <div className="text-muted-foreground">Target NPU</div>
            <div className="font-medium">{hardwareInfo?.npu?.name ?? "AMD NPU (auto)"}</div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge variant={compatible ? "success" : "warning"}>
            {compatible ? "safetensors compatible" : "check format before NPU compile"}
          </Badge>
          {npuAvailable === true && <Badge variant="success">NPU detected</Badge>}
          {npuAvailable === false && <Badge variant="destructive">NPU not detected</Badge>}
          {npuAvailable === null && <Badge variant="secondary">hardware unknown</Badge>}
          <Badge variant="outline">EP: {targetEp}</Badge>
          <Badge variant="outline">
            {precision} / {device} / {powerMode}
          </Badge>
        </div>

        <div className="rounded-lg bg-muted p-3 text-sm">
          <span className="font-medium">Recommended config: </span>
          {recommendedPrecision(modelMetadata.params)} — current selection: {precision} on {device}{" "}
          via {targetEp}.
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Conversion failed to start</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Button onClick={() => mutation.mutate()} disabled={mutation.isPending} size="lg">
          {mutation.isPending && <Loader2 className="animate-spin" />}
          Convert to NPU
        </Button>
      </CardContent>
    </Card>
  );
}
