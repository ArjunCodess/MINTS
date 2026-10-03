"""Run an argv command and append its exact exit status and logs to a ledger."""
from pathlib import Path
import sys
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
out = root / "results/review"
out.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
ledger = out / "commands.jsonl"
log = out / f"command_{uuid.uuid4().hex[:12]}.log"
started = time.perf_counter()
with log.open("w", encoding="utf-8") as stream:
    result = subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
record = dict(argv=args, cwd=str(root), exit_status=result.returncode, seconds=time.perf_counter()-started,
              timestamp=datetime.now(timezone.utc).isoformat(), log=str(log.relative_to(root)))
with ledger.open("a",encoding="utf-8") as stream:
    stream.write(json.dumps(record)+"\n")
print(json.dumps(record),flush=True)
print(log.read_text(encoding="utf-8",errors="replace")[-6000:],flush=True)
sys.exit(result.returncode)
