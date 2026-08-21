# Q/R/SKM evidence catalogue

The evidence collector scans completed event files from Q and R runs plus the
shared single-knot-mastery witness inventory. It never writes into experiment
directories. Its only writable target is a dedicated catalogue directory.

Admission has two independent gates:

1. the exact semantic witness must replay from its recorded braid word to the
   empty one-braid;
2. a knot-level upper bound is exported only when the declared knot name and
   computed invariant fingerprint are compatible.

The exact representation and replayable witness remain useful even when knot
identification is ambiguous. Conflicting names, malformed rows, and witnesses
that fail replay are kept in `quarantine.jsonl`; they are not silently dropped
or promoted. Source event, bank, manifest, protocol, and content hashes are
stored with each record.

`evidence.sqlite3` is the one canonical append-only database. Science-specific
databases are represented as reproducible collections inside it, rather than
independent writable copies that can drift. Collections are generated for each
experiment, source family, and exact solver version. `science-collections.json`
is their compact index.

Every admitted solution has two normalized identities in addition to its raw
manifest provenance:

- a solver version: architecture/scientist, checkpoint path and SHA-256, and
  source/hash manifests;
- a search protocol: simulations, action horizon, objective, seed, evaluation
  attempts, and other available search parameters.

`best-solutions-pool.json` contains the replay-verified global best per exact
representation and knot, plus the best result per representation and solver
version. It includes the replayable witness and normalized solver/search
metadata. `solver-versions.json` and `search-protocols.json` are inspection
indexes. The legacy `best-knot-upper-bounds.json` remains available.

The action alphabet is evidence-specific. Every admitted solution records
`cyclic_band_generators`, and the summary separates ordinary-Artin from B*
evidence. The same starting braid may therefore have both ordinary and B*
witnesses without an instance-ID collision. Q/R flat-event replay reads the
flag from the frozen run manifest; mastery and external witnesses carry it in
their serialized action specification. Existing witnesses are never
reinterpreted when a B* scientist is added.

External records are ingested into a separate tier. A sourced upper-bound claim
without a full replayable action path is retained as `claim-only`, never mixed
into replay-verified rankings, and never marked eligible for distillation.
Future external paths may enter the verified pool only after the same native
replay gate as our own results. This keeps external knowledge useful for
challenge selection without turning a citation into training evidence.

`representation-metadata.jsonl` and `quarantine.jsonl` remain atomic derived
views for inspection and backup.

Metadata is incremental. Alexander polynomial, determinant, signature (when
Spherogram is available), genus bounds, and the signature lower bound are
computed for every representation. Jones/table identification is computed up
to the configured strand limit (eight by default); wider representations are
explicitly marked `partial-capacity` so an expensive 9--12-strand Jones
calculation cannot stall ingestion.

## Service configuration

Install `deploy/systemd/evidence-metadata-collector.{service,timer}` and provide
`/etc/braid/evidence-metadata-collector.env`. Root lists are colon-separated:

```bash
EVIDENCE_COLLECTOR_EXPERIMENT_ROOTS=/srv/braid/artifacts/semantic-v2-r200-optimized:/srv/braid/artifacts/q4000-strand12-20260814:/srv/braid/artifacts/q4000-strand12-fresh-v1-20260815
EVIDENCE_COLLECTOR_MASTERY_INVENTORIES=/srv/braid/artifacts/multi-knot-mastery-v1-20260815/evidence-inventory
EVIDENCE_COLLECTOR_EXTERNAL_EVIDENCE=/srv/braid/artifacts/evidence-catalog-v1/inputs/external-unknotting-evidence-20260815.json:/srv/braid/artifacts/evidence-catalog-v1/inputs/external-upper-bound-collection-20260815.json
EVIDENCE_COLLECTOR_OUTPUT=/srv/braid/artifacts/evidence-catalog-v1
EVIDENCE_COLLECTOR_METADATA_LIMIT=8
EVIDENCE_COLLECTOR_MAX_FULL_STRANDS=8
```

The timer invokes a low-priority oneshot every 20 minutes. A non-blocking file
lock prevents duplicate collectors. Repeated scans are idempotent by source
hash and evidence identity.
