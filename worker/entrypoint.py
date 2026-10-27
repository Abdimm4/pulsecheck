"""PulseCheck background worker and cleanup command.

Usage (from the repo root, with the backend installed: pip install -e "backend[api]"):
    python worker/entrypoint.py worker            # process analyses until stopped
    python worker/entrypoint.py worker --burst    # process what is queued, then exit
    python worker/entrypoint.py cleanup           # one expiry/deletion sweep (UP-4)
    python worker/entrypoint.py cleanup --loop 300

Settings come from configs/default.yaml and the environment (DATABASE_URL,
REDIS_URL, ...; see .env.example). The worker is RQ's SimpleWorker, which
does not fork and therefore also runs on Windows; run several processes for
parallel analyses. Logs carry job IDs, never access tokens.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from pulsecheck.jobs import tasks
from pulsecheck.jobs.cleanup import run_cleanup
from pulsecheck.jobs.queue import make_worker
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.settings import Settings

log = logging.getLogger("pulsecheck.worker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker", help="run the analysis worker")
    work.add_argument("--burst", action="store_true", help="exit when the queue is empty")
    clean = sub.add_parser("cleanup", help="expire jobs and delete files")
    clean.add_argument("--loop", type=float, metavar="SECONDS", help="repeat every SECONDS")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    runtime = Runtime.from_settings(Settings.load())
    tasks.set_runtime(runtime)

    if args.command == "worker":
        # Build the pipeline (and load any trained model) before taking jobs.
        log.info("model: %s %s", *runtime.model())
        runtime.analyzer  # noqa: B018 - eager load
        worker = make_worker(runtime.redis)
        worker.work(burst=args.burst, with_scheduler=False)
        return 0

    while True:
        run_cleanup(runtime)
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
