#!/bin/bash
# Download the sealed held-out outputs from Zenodo and check their sha256 before use.
set -euo pipefail
DOI_URL=${ZENODO_URL:?set ZENODO_URL to the Zenodo record's file base URL (DOI: to be assigned)}
D=${REPRO_DATA:-data}; mkdir -p "$D/edm-v2-e0" "$D/edm-v2-e0b"
for f in e0-heldout.tar e0b-heldout.tar; do curl -fsSL -o "$D/$f" "$DOI_URL/$f"; done
cd "$D" && printf '%s  %s\n%s  %s\n' bd06a11eb7163da102cad6ea0df12ea79e52ff94aab1dc35d28ec3910af2be22 e0-heldout.tar 68219f228c41da972fe4c8f92c218e01f85f08eed864185f237cecda05314bbf e0b-heldout.tar | shasum -a 256 -c
tar -xf e0-heldout.tar -C edm-v2-e0 && tar -xf e0b-heldout.tar -C edm-v2-e0b
printf '%s  %s\n%s  %s\n' aff9a8c49d48821c0e9343cd058c4e0e2f1659ffabcaa719824412f3b14abc90 edm-v2-e0/heldout/summary.json 617be427feac24df2e342da00d34e91c78d5de91592cee390ef3516ae9e2ff3f edm-v2-e0b/heldout/summary.json | shasum -a 256 -c
