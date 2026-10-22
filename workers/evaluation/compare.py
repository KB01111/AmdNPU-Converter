"""Compare original vs quantized ONNX: size, latency proxy, numeric drift.

    python workers/evaluation/compare.py --baseline base.onnx --candidate quant.onnx [--eval-npz eval.npz] [--max-degradation 0.02] [--latency-ms ... ]

Exit non-zero when degradation exceeds threshold (honest gate for CI).
Numeric drift measured on --eval-npz (keys = input names) or synthetic ones().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare baseline vs quantized ONNX.")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--eval-npz", default=None, help=".npz with arrays per input name")
    p.add_argument("--max-degradation", type=float, default=0.02, help="Max allowed relative drift, 0.02 means 2 percent")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args(argv)


def run_model(path: str, feed: dict):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feed), [{"name": i.name} for i in sess.get_inputs()]


def main(argv=None) -> int:
    args = parse_args(argv)
    for k in ("baseline", "candidate"):
        if not Path(getattr(args, k)).exists():
            print(json.dumps({"error": f"not found: {getattr(args, k)}"}), file=sys.stderr)
            return 2
    try:
        import numpy as np
    except Exception as e:
        print(json.dumps({"error": f"numpy missing: {e}"}), file=sys.stderr)
        return 1

    try:
        import onnxruntime as ort
        s0 = ort.InferenceSession(args.baseline, providers=["CPUExecutionProvider"])
        metas = [(i.name, list(i.shape), i.type) for i in s0.get_inputs()]
    except Exception as e:
        print(json.dumps({"error": f"baseline open failed: {e}"}), file=sys.stderr)
        return 1

    if args.eval_npz and Path(args.eval_npz).exists():
        try:
            z = np.load(args.eval_npz)
            feed = {m[0]: np.asarray(z[m[0]]) for m in metas if m[0] in z}
            if len(feed) != len(metas):
                print(json.dumps({"error": f"eval npz missing inputs (have {list(z.files)}, need {[m[0] for m in metas]})"}),
                      file=sys.stderr)
                return 2
            source = "eval-npz"
        except Exception as e:
            print(json.dumps({"error": f"eval npz load failed: {e}"}), file=sys.stderr)
            return 1
    else:
        feed = {}
        for name, shape, typ in metas:
            shp = []
            for idx, d in enumerate(shape):
                shp.append(d if isinstance(d, int) else (args.batch if idx == 0 else args.seq_len))
            feed[name] = np.ones(shp, dtype=np.int64 if "int" in str(typ) else np.float32)
        source = "synthetic-ones"
        print(json.dumps({"warning": "no --eval-npz; drift measured on synthetic ones() input"}), file=sys.stderr)

    try:
        b_out, _ = run_model(args.baseline, feed)
        c_out, _ = run_model(args.candidate, feed)
    except Exception as e:
        print(json.dumps({"error": f"inference failed: {e}"}), file=sys.stderr)
        return 1

    import math
    drifts = []
    for b, c in zip(b_out, c_out):
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        if b.shape != c.shape:
            drifts.append({"shape_mismatch": [list(b.shape), list(c.shape)], "rel": float("inf")})
            continue
        denom = float(np.mean(np.abs(b))) or 1.0
        rel = float(np.mean(np.abs(b - c)) / denom)
        drifts.append({"mean_abs_err": float(np.mean(np.abs(b - c))),
                       "max_abs_err": float(np.max(np.abs(b - c))),
                       "rel_drift": rel,
                       "allclose": bool(np.allclose(b, c, rtol=1e-2, atol=1e-3))})
    worst = max((d.get("rel_drift", float("inf")) for d in drifts), default=float("inf"))
    bb = Path(args.baseline).stat().st_size
    cb = Path(args.candidate).stat().st_size
    rep = {"baseline": args.baseline, "candidate": args.candidate, "eval_source": source,
           "baseline_bytes": bb, "candidate_bytes": cb,
           "size_ratio": cb / bb if bb else None,
           "per_output": drifts, "worst_rel_drift": worst,
           "threshold": args.max_degradation,
           "pass": bool(worst <= args.max_degradation)}
    print(json.dumps(rep))
    return 0 if rep["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
