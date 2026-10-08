# Mode-B server snapshot

Original code, results, checkpoints and tags are unchanged.
Large JSON copies remove formatting whitespace only; all parsed fields are preserved.
a1_comparison_overview.json omits channel rows; the full eval JSON retains them.
training_overview.json reports history minima, not newly selected checkpoints.
Use saved runtime config/arguments, not YAML defaults alone.
This package contains no weights, replay tensors, ZIPs, or external A1 source.
Replay manifest describes server-local tensor files, not files included here.
Historical provenance remains unchanged; snapshot code identity is in snapshot_index.json.
