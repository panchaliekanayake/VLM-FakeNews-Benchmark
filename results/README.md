# Published result provenance

The two CSV files in this directory are machine-readable transcriptions of Tables III and IV in the accompanying manuscript. They are committed so readers can inspect the complete reported values and regenerate the trade-off figure.

They are not recomputed during CI and should not be presented as independently reproduced results. The original full prediction artifacts are not committed: they are too large for a normal Git repository and contain dataset-derived text and paths whose redistribution may be restricted. New runs write complete predictions, metrics, and provenance under `outputs/`; those files can be scored with `litevlm-fnd evaluate`.

The release implementation preserves the paper prompt, parser behavior, fixed-class macro scoring, seed, batch size, and timing rule. It is a cleaned and modularized release implementation assembled from the experimental code, not a byte-for-byte snapshot of the research working directory. Consequently, users should expect dependency and hardware differences to affect both generated answers and latency.

For an archival release, upload full prediction JSON files as separately access-controlled artifacts only if every source dataset's redistribution terms permit it. Record their SHA-256 digests and link that immutable archive here after it exists; do not add a speculative URL.
