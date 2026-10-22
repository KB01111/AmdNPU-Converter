import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { inspectHfModel } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";

const HF_URL_RE = /^https:\/\/huggingface\.co\/([A-Za-z0-9._-]+)\/([A-Za-z0-9._-]+)\/?$/;

export function validateHfUrl(url: string): { owner: string; repo: string } | null {
  const m = url.trim().match(HF_URL_RE);
  if (!m) return null;
  return { owner: m[1], repo: m[2] };
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
}

export default function HuggingFaceImport() {
  const huggingfaceUrl = useAppStore((s) => s.huggingfaceUrl);
  const setHuggingfaceUrl = useAppStore((s) => s.setHuggingfaceUrl);
  const setModelMetadata = useAppStore((s) => s.setModelMetadata);
  const modelMetadata = useAppStore((s) => s.modelMetadata);
  const [validationError, setValidationError] = React.useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: (url: string) => inspectHfModel(url),
    onSuccess: (data) => {
      setModelMetadata(data);
      setValidationError(null);
    },
  });

  const handleInspect = () => {
    const parsed = validateHfUrl(huggingfaceUrl);
    if (!parsed) {
      setValidationError("URL must match https://huggingface.co/<owner>/<repo>");
      return;
    }
    setValidationError(null);
    mutation.mutate(huggingfaceUrl.trim());
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Import from Hugging Face</CardTitle>
          <CardDescription>
            Paste a model URL, e.g. https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              value={huggingfaceUrl}
              onChange={(e) => setHuggingfaceUrl(e.target.value)}
              placeholder="https://huggingface.co/<owner>/<repo>"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleInspect();
              }}
            />
            <Button onClick={handleInspect} disabled={mutation.isPending || !huggingfaceUrl.trim()}>
              {mutation.isPending ? (
                <Loader2 className="animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              Inspect
            </Button>
          </div>
          {validationError && (
            <Alert variant="destructive">
              <AlertTitle>Invalid URL</AlertTitle>
              <AlertDescription>{validationError}</AlertDescription>
            </Alert>
          )}
          {mutation.isError && (
            <Alert variant="destructive">
              <AlertTitle>Inspection failed</AlertTitle>
              <AlertDescription>
                {mutation.error instanceof Error ? mutation.error.message : String(mutation.error)}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {modelMetadata && (
        <Card>
          <CardHeader>
            <CardTitle>{modelMetadata.modelId}</CardTitle>
            <CardDescription>Metadata returned by hf_inspect</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-3">
              <div>
                <div className="text-muted-foreground">Architecture</div>
                <div className="font-medium">{modelMetadata.arch}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Parameters</div>
                <div className="font-medium">
                  {modelMetadata.paramsHuman} ({modelMetadata.params.toLocaleString()})
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">Format</div>
                <div className="font-medium">{modelMetadata.format}</div>
              </div>
              <div>
                <div className="text-muted-foreground">License</div>
                <div className="font-medium">{modelMetadata.license ?? "unknown"}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Total size</div>
                <div className="font-medium">{formatBytes(modelMetadata.totalSizeBytes)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Files</div>
                <div className="font-medium">{modelMetadata.files.length}</div>
              </div>
            </div>

            {modelMetadata.safetensors && (
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge variant="secondary">
                  safetensors: {modelMetadata.safetensors.numTensors} tensors
                </Badge>
                {modelMetadata.safetensors.dtype && (
                  <Badge variant="outline">{modelMetadata.safetensors.dtype}</Badge>
                )}
                <Badge variant="outline">
                  {modelMetadata.safetensors.totalParams.toLocaleString()} params
                </Badge>
              </div>
            )}

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="py-2 pr-4 font-medium">File</th>
                    <th className="py-2 pr-4 font-medium">Size</th>
                    <th className="py-2 font-medium">SHA-256</th>
                  </tr>
                </thead>
                <tbody>
                  {modelMetadata.files.map((f) => (
                    <tr key={f.name} className="border-b last:border-0">
                      <td className="py-1.5 pr-4 font-mono text-xs">{f.name}</td>
                      <td className="py-1.5 pr-4">{formatBytes(f.sizeBytes)}</td>
                      <td className="py-1.5 font-mono text-xs text-muted-foreground">
                        {f.sha256 ? `${f.sha256.slice(0, 16)}…` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
