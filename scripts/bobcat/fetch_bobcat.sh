#!/bin/bash
# Fetch BOBCAT (MIT, github.com/arghosh/BOBCAT) at the pinned commit, which also carries the preprocessed ASSISTments
# 2009-2010 file. The ASSISTments terms of use forbid passing the data on, so this repo never contains it; you fetch it
# from its source and accept those terms yourself (sites.google.com/site/assistmentsdata/termsofuseforusingdata).
set -euo pipefail
D=${REPRO_DATA:-data}/published-cat-audit-v1
COMMIT=e6b6245e23c1065ba8b8c56de5f051dfbcdd89ec
WANT=c07ca9fa0f08207b06488cec527725938fe96e81e3c1762839e1ad0b84f974cc
mkdir -p "$D"
[ -d "$D/arghosh-BOBCAT/.git" ] || git clone -q https://github.com/arghosh/BOBCAT "$D/arghosh-BOBCAT"
git -C "$D/arghosh-BOBCAT" checkout -q "$COMMIT"
F="$D/arghosh-BOBCAT/data/train_task_assist2009.json"
GOT=$(shasum -a 256 "$F" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "ASSISTments file hash mismatch: $GOT"; exit 1; }
python3 - "$D" "$F" "$WANT" <<'PY'
import json, sys, pathlib
d, f, sha = sys.argv[1:]
pathlib.Path(d, 'dataset-receipt.json').write_text(json.dumps(dict(file=str(pathlib.Path(f).resolve()), sha256=sha, source_commit='e6b6245e23c1065ba8b8c56de5f051dfbcdd89ec'), indent=1))
PY
echo "ok: BOBCAT $COMMIT and ASSISTments 2009 file verified"
