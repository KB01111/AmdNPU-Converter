import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { useAppStore } from "@/stores/useAppStore";
import type { DevicePreference, PowerMode, Precision, TargetEp } from "@/lib/types";

export default function ConversionSettings() {
  const precision = useAppStore((s) => s.precision);
  const device = useAppStore((s) => s.device);
  const powerMode = useAppStore((s) => s.powerMode);
  const targetEp = useAppStore((s) => s.targetEp);
  const setPrecision = useAppStore((s) => s.setPrecision);
  const setDevice = useAppStore((s) => s.setDevice);
  const setPowerMode = useAppStore((s) => s.setPowerMode);
  const setTargetEp = useAppStore((s) => s.setTargetEp);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Conversion settings</CardTitle>
        <CardDescription>Precision, device, power mode and execution provider.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Precision</label>
          <Select
            value={precision}
            onValueChange={(v) => setPrecision(v as Precision)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "fp32", label: "FP32" },
              { value: "bf16", label: "BF16" },
              { value: "a16w8", label: "A16W8 (NPU)" },
              { value: "a8w8", label: "A8W8 (NPU)" },
              { value: "int8", label: "INT8" },
            ]}
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Device</label>
          <Select
            value={device}
            onValueChange={(v) => setDevice(v as DevicePreference)}
            options={[
              { value: "auto", label: "Auto" },
              { value: "npu", label: "NPU" },
              { value: "gpu", label: "GPU" },
              { value: "cpu", label: "CPU" },
            ]}
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Power mode</label>
          <Select
            value={powerMode}
            onValueChange={(v) => setPowerMode(v as PowerMode)}
            options={[
              { value: "automatic", label: "Automatic" },
              { value: "max-performance", label: "Max performance" },
              { value: "battery-efficient", label: "Battery efficient" },
            ]}
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Target execution provider</label>
          <Select
            value={targetEp}
            onValueChange={(v) => setTargetEp(v as TargetEp)}
            options={[
              { value: "vitisai", label: "VitisAI (AMD NPU)" },
              { value: "qnn", label: "QNN (Qualcomm)" },
              { value: "cuda", label: "CUDA (NVIDIA)" },
              { value: "cpu", label: "CPU" },
            ]}
          />
        </div>
      </CardContent>
    </Card>
  );
}
