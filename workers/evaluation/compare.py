"""Compare original vs quantized ONNX: size and numeric drift.

    python workers/evaluation/compare.py --baseline base.onnx --candidate quant.onnx --eval-npz eval.npz --max-degradation 0.02

Exit non-zero when degradation exceeds threshold (honest gate for CI).
Numeric drift needs representative --eval-npz (keys = input names). Without it,
synthetic smoke results are reported but cannot pass the acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import math
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
    return sess.run(None, feed), [{"name": i.name} for i in sess.get_outputs()]


def main(argv=None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.max_degradation) or args.max_degradation < 0 or args.batch < 1 or args.seq_len < 1:
        print(json.dumps({"error": "finite nonnegative max-degradation and positive batch/seq-len required"}), file=sys.stderr)
        return 2
    for k in ("baseline", "candidate"):
        if not Path(getattr(args, k)).is_file():
            print(json.dumps({"error": f"not found: {getattr(args, k)}"}), file=sys.stderr)
            return 2
    if args.eval_npz and not Path(args.eval_npz).is_file():
        print(json.dumps({"error": f"eval npz not found: {args.eval_npz}"}), file=sys.stderr)
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

    if args.eval_npz:
        try:
            with np.load(args.eval_npz, allow_pickle=False) as z:
                feed = {m[0]: np.asarray(z[m[0]]) for m in metas if m[0] in z}
                if len(feed) != len(metas):
                    raise ValueError(f"missing inputs (have {list(z.files)}, need {[m[0] for m in metas]})")
            if any(arr.size == 0 or not np.isfinite(arr).all() for arr in feed.values()):
                raise ValueError("evaluation inputs must be nonempty and finite")
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
        b_out, b_meta = run_model(args.baseline, feed)
        c_out, c_meta = run_model(args.candidate, feed)
    except Exception as e:
        print(json.dumps({"error": f"inference failed: {e}"}), file=sys.stderr)
        return 1

    if not b_out or len(b_out) != len(c_out):
        print(json.dumps({"error": f"output count mismatch or empty outputs: baseline={len(b_out)}, candidate={len(c_out)}"}), file=sys.stderr)
        return 3
    if b_meta != c_meta:
        print(json.dumps({"error": "baseline/candidate output names or order differ"}), file=sys.stderr)
        return 3
    drifts = []
    for b, c in zip(b_out, c_out):
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        if b.shape != c.shape:
            print(json.dumps({"error": "output shape mismatch", "shapes": [list(b.shape), list(c.shape)]}), file=sys.stderr)
            return 3
        if b.size == 0 or not np.isfinite(b).all() or not np.isfinite(c).all():
            print(json.dumps({"error": "outputs must be nonempty and finite"}), file=sys.stderr)
            return 3
        denom = float(np.mean(np.abs(b))) or 1.0
        if not math.isfinite(denom):
            print(json.dumps({"error": "nonfinite reference magnitude"}), file=sys.stderr)
            return 3
        rel = float(np.mean(np.abs(b - c)) / denom)
        if not math.isfinite(rel):
            print(json.dumps({"error": "nonfinite output drift"}), file=sys.stderr)
            return 3
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
           "pass": bool(source == "eval-npz" and worst <= args.max_degradation)}
    print(json.dumps(rep, allow_nan=False))
    return 0 if rep["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
