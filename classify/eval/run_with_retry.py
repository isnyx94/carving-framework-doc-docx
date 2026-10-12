"""Supervisor for long checkpointed runs that keep dying silently.

Why this exists
---------------
The Chapter 5 4,096-byte experiment has now died four separate times — twice to
session teardown, once killed manually to fix a label confound, and once silently
with an empty stderr and no traceback. Each death cost hours, and one went
unnoticed for three weeks. The stage scripts are all checkpointed, so a relaunch
skips completed work; the missing piece was something to notice the death and do
the relaunch.

This wrapper runs a module repeatedly until a **sentinel file** appears (the
artefact the run produces only on success) or an attempt budget is exhausted. It
makes no assumption that the child exited cleanly: a silent death, a non-zero
exit and a hang-then-kill all look the same from here, and all are handled by
trying again.

Progress-stall detection is included because a hung run is worse than a dead one
— a dead run at least stops consuming the GPU. If the checkpoint file has not
changed size or mtime within `--stall-minutes`, the child is killed and retried.

Usage:
    python -m classify.eval.run_with_retry \
        --module classify.eval.ch5_resume \
        --sentinel results/ch5/CH5_REPORT.md \
        --checkpoint results/ch5/compound_partial.json \
        --log results/ch5_supervised.log \
        --attempts 12 --stall-minutes 90
"""

import argparse
import os
import subprocess
import sys
import time


def _stat(path):
    try:
        st = os.stat(path)
        return (st.st_size, int(st.st_mtime))
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="module to run with -m")
    ap.add_argument("--sentinel", required=True,
                    help="file that exists only when the run has succeeded")
    ap.add_argument("--checkpoint", default=None,
                    help="file whose growth indicates progress; enables stall detection")
    ap.add_argument("--log", required=True, help="combined child stdout/stderr")
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--stall-minutes", type=int, default=90)
    ap.add_argument("--cooldown-seconds", type=int, default=20,
                    help="pause between attempts, to let GPU memory be released")
    ap.add_argument("--wait-for", default=None,
                    help="block until this file exists before the first attempt. Used to "
                         "chain GPU jobs: the card is compute-saturated by one training "
                         "run, so overlapping a second only adds contention and OOM risk "
                         "to a run that has already died repeatedly.")
    ap.add_argument("--wait-timeout-hours", type=float, default=24.0)
    args = ap.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    if args.wait_for:
        deadline = time.time() + args.wait_timeout_hours * 3600
        print(f"[supervisor] waiting for {args.wait_for} before starting "
              f"(timeout {args.wait_timeout_hours}h)", flush=True)
        while not os.path.exists(args.wait_for):
            if time.time() > deadline:
                print(f"[supervisor] timed out waiting for {args.wait_for}; "
                      f"starting anyway", flush=True)
                break
            time.sleep(60)
        else:
            print(f"[supervisor] {args.wait_for} appeared; proceeding", flush=True)

    for attempt in range(1, args.attempts + 1):
        if os.path.exists(args.sentinel):
            print(f"[supervisor] sentinel {args.sentinel} present; nothing to do", flush=True)
            return 0

        print(f"[supervisor] attempt {attempt}/{args.attempts}: "
              f"python -m {args.module}", flush=True)
        with open(args.log, "a", encoding="utf-8") as log:
            log.write(f"\n===== supervisor attempt {attempt} "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log.flush()
            proc = subprocess.Popen([sys.executable, "-m", args.module],
                                    stdout=log, stderr=subprocess.STDOUT, env=env)

            last = _stat(args.checkpoint) if args.checkpoint else None
            last_change = time.time()
            while proc.poll() is None:
                time.sleep(30)
                if not args.checkpoint:
                    continue
                current = _stat(args.checkpoint)
                if current != last:
                    last, last_change = current, time.time()
                elif time.time() - last_change > args.stall_minutes * 60:
                    print(f"[supervisor] no checkpoint progress for "
                          f"{args.stall_minutes} min; killing and retrying", flush=True)
                    proc.kill()
                    break
            code = proc.wait()

        if os.path.exists(args.sentinel):
            print(f"[supervisor] SUCCESS on attempt {attempt} "
                  f"(child exit {code})", flush=True)
            return 0
        print(f"[supervisor] attempt {attempt} ended without sentinel "
              f"(child exit {code}); retrying after cooldown", flush=True)
        time.sleep(args.cooldown_seconds)

    print(f"[supervisor] exhausted {args.attempts} attempts without producing "
          f"{args.sentinel}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
