# App completion verification - 2026-10-22

## Implemented scope

Completed the approved existing-product scope: command/DTO alignment, HF/local
ONNX/CUA-S1 import, settings and data-only recipes, safe worker routing, numeric
gates, process cancellation/restart/retry, hardware readiness, benchmark/provider
reporting, producing-artifact actions, accessibility, resource packaging, and CI.

An independent review identified six integration defects. All were corrected and
re-reviewed: app-exit descendants, selected benchmark provider, empty optional
calibration on save, artifact/job provenance, preset relative output defaults,
and the local preparation stage label. WindowsML runtime namespace separation
and dynamic provider selection were also corrected and covered by fixtures.

## Local evidence

| Check | Result | Acceptance boundary |
|---|---|---|
| `npm test` | 23 passed in five files | frontend contracts/state/component behavior |
| `npx playwright test` | 6 passed in serial installed Edge and CI-mode Chromium | explicit browser native fixtures; no page errors |
| `npm run build` | passed | TypeScript and Vite; entry bundle about 241 kB vs baseline 549 kB |
| `cargo test --locked --manifest-path src-tauri/Cargo.toml` | 27 passed, no ignored tests | serde/routing/storage plus real lightweight Windows process-tree tests |
| `cargo check` and `cargo fmt` | passed | native compilation/formatting |
| Python 3.11 unittest discovery | 39 passed | external ML SDKs are explicit fakes |
| Python 3.11 compileall | passed | syntax |
| `npm audit --omit=dev` | zero production advisories | full audit retains seven existing Tailwind v3 development-tree advisories |
| `npm run tauri -- build --debug --bundles nsis` | passed | native debug executable and 6.07 MiB NSIS installer built |
| Bundled resources | 25 files matched source hashes | includes Python/scripts/setup data and recipes, excludes virtual environments |

The built native executable was launched and observed through the Windows UI.
Import/settings rendered, Rust supplied seven supported recipe choices, and
Hardware displayed real CPU/GPU/NPU and driver information. Missing WinML and
the absent isolated Python environment appeared as setup errors while retaining
the hardware facts. The test app was then closed. The normal repository checkout
was left clean; implementation is isolated on the PR branch/worktree.

## Unverified acceptance

No ML environments, drivers, providers, model weights, or calibration datasets
were installed/downloaded for this verification. Real HF/ONNX/CUA model export,
Quark quantization, WinML compilation, representative numeric acceptance, dynamic
WindowsML execution, and live GPU/NPU placement remain unverified. CUA additionally
requires its trusted official source package. The installer was built but not
installed; release packaging and signed distribution were not exercised.

Do not interpret fixture tests, device detection, or provider names as hardware
acceptance. Use tests/e2e-conversion.md with actual models/data/tool versions.
Existing virtual environments containing conflicting ORT distributions need to
be recreated according to workers/README.md.

Hosted CI is recorded on the pull request after publication; local checks alone
do not establish a hosted result.

The first hosted UI run caught a test-selector race: a broad final Benchmark
button locator could select main navigation before the asynchronous model
actions loaded. The test now waits for the exact Benchmark compiled action;
all six cases passed locally in CI-mode Chromium after this correction.

Hosted review also exposed HF suffix-link validation mismatches and overflowing
parameter metadata. Added regressions cover canonical recipe saves for HF
tree/blob/resolve links, invalid path rejection, and invalid parameter counts.
Two historical recipe annotations now accurately state that Auto selects BF16.
