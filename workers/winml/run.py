"""WinML CLI pass-through wrapper.

Finds winml.exe (env WINML_BIN, workers/.venv*/Scripts/winml.exe, PATH)
and forwards subcommand args via subprocess with streaming.

    python workers/winml/run.py <inspect|export|optimize|quantize|compile|build|perf|eval|config|sys> [args...]
    python workers/winml/run.py sys --output-json [--job-id ID]

--job-id: emit JSONL progress lines to stdout for the Rust parser, while
forwarding child stderr/stdout through.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SUBCOMMANDS = ["inspect", "export", "optimize", "quantize", "compile", "build", "perf", "eval", "config", "sys"]

WORKERS_DIR = Path(__file__).resolve().parents[1]


def find_winml_bin(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("WINML_BIN")
    if env and Path(env).exists():
        return env
    for venv in (".venv-winml", ".venv"):
        cand = WORKERS_DIR / venv / "Scripts" / "winml.exe"
        if cand.exists():
            return str(cand)
        cand2 = WORKERS_DIR / venv / "bin" / "winml"
        if cand2.exists():
            return str(cand2)
    found = shutil.which("winml.exe") or shutil.which("winml")
    return found


def emit_progress(job_id: str, phase: str, **fields):
    rec = {"job_id": job_id, "phase": phase, "ts": time.time()}
    rec.update(fields)
    print(json.dumps(rec), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="winml.exe pass-through wrapper.")
    ap.add_argument("subcommand", choices=SUBCOMMANDS, help="winml subcommand")
    ap.add_argument("--winml-bin", dest="winml_bin", default=None, help="Explicit winml.exe path")
    ap.add_argument("--job-id", dest="job_id", default=None, help="Emit JSONL progress for Rust parser")
    ap.add_argument("--output-json", dest="output_json", action="store_true", help="Pass --output-json to winml if supported")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="Args forwarded to winml.exe")
    args = ap.parse_args(argv)

    exe = find_winml_bin(args.winml_bin)
    if not exe:
        print(json.dumps({"error": "winml.exe not found. Set WINML_BIN, install requirements-winml.txt, or put winml.exe on PATH."}),
              file=sys.stderr)
        return 2

    fwd = [s for s in args.rest if s != "--"]
    cmd = [exe, args.subcommand] + fwd
    # Only add --output-json when requested AND not already present (some subcommands may not support it;
    # we pass through honestly and let winml.exe decide).
    if args.output_json and "--output-json" not in fwd and "--json" not in fwd:
        cmd.append("--output-json")

    if args.job_id:
        emit_progress(args.job_id, "start", cmd=cmd, exe=exe, subcommand=args.subcommand)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except FileNotFoundError:
        print(json.dumps({"error": f"failed to launch: {exe}"}), file=sys.stderr)
        return 1
    except Exception as e:
        print(json.dumps({"error": f"launch failed: {e}"}), file=sys.stderr)
        return 1

    assert proc.stdout is not None
    for line in proc.stdout:
        # stream child output verbatim so Rust parser sees it
        sys.stdout.write(line)
        sys.stdout.flush()
        if args.job_id and line.strip():
            emit_progress(args.job_id, "log", line=line.rstrip("\n"))
    rc = proc.wait()
    if args.job_id:
        emit_progress(args.job_id, "done", returncode=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
