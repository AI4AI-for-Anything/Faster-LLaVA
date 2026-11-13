"""CPU check of gate_eval.py's decision rule on synthetic paired-seed trajectories.

Locks the verdicts the LLaVA gate relies on, with the defaults used there
(skip 10 steps, 5 endpoints, baseline arm "up", 3 seeds):
  - a candidate whose paired differences are far inside delta is PASS;
  - a candidate shifted by 0.0015, inside delta = 2 x baseline seed std (0.002) but
    outside 1 x std, is PASS (locks the factor 2);
  - a candidate shifted by 0.05 at every step (delta here is ~0.002) is INCORRECT;
  - a candidate outside delta whose paired differences change sign across seeds
    (no significant paired effect) is UNRESOLVED.
Input goes through a pipe in the exact line format gate_eval parses, so no file
is written.

Run from the repository root (pure Python, no torch):
  python -m tpu_smoke.test_gate_eval
"""
import contextlib
import io
import json
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from tpu_smoke import gate_eval  # noqa: E402

STEPS = 30
SEED_OFFSET = {0: 0.0, 1: 0.001, 2: -0.001}     # baseline seed spread -> delta ~ 0.002


def traj(seed, shift):
    return [[5.0 - 0.02 * t + SEED_OFFSET[seed] + shift(seed, t), None] for t in range(STEPS)]


ARMS = {
    "up": lambda s, t: 0.0,
    "near": lambda s, t: 1e-5 * (s - 1),                      # PASS
    "edge": lambda s, t: 0.0015 + 1e-5 * (s - 1),             # PASS only with delta = 2 x std
    "shift": lambda s, t: -0.05 + 1e-4 * s,                   # INCORRECT
    "noisy": lambda s, t: {0: 0.05, 1: -0.01, 2: 0.02}[s],    # UNRESOLVED: mean 0.02, p ~ 0.4
}


def run_gate():
    lines = "".join(f"w0 {arm} s{seed} TRAJ " + json.dumps(dict(seed=seed, traj=traj(seed, f))) + "\n"
                    for arm, f in ARMS.items() for seed in SEED_OFFSET)
    r, w = os.pipe()

    def feed():
        with os.fdopen(w, "w") as fh:
            fh.write(lines)
    th = threading.Thread(target=feed)
    th.start()
    out = io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["gate_eval", f"/dev/fd/{r}"]
        with contextlib.redirect_stdout(out):
            gate_eval.main()
    finally:
        sys.argv = argv
        th.join()
        os.close(r)
    verdicts = {}
    for line in out.getvalue().splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0] in ARMS:
            verdicts[parts[0]] = parts[2]
    return verdicts, out.getvalue()


def main():
    verdicts, text = run_gate()
    want = {"near": "PASS", "edge": "PASS", "shift": "INCORRECT", "noisy": "UNRESOLVED"}
    for arm, v in want.items():
        assert verdicts.get(arm) == v, f"{arm}: expected {v}, got {verdicts.get(arm)}\n{text}"
        print(f"PASS {arm} -> {v}")


if __name__ == "__main__":
    main()
