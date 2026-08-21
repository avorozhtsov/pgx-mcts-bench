# Additive Q4000 B* capacity-recovery candidate

This is a fresh candidate, not a flag change to any running scientist.

The active Q and SKM checkpoints were trained with
`cyclic_band_generators=false`. Enabling B* changes the observation alphabet,
legal-action set, and policy head. Existing states and evidence therefore stay
immutable and ordinary-Artin.

The additive candidate uses the scalable torus raster with a shared adjacent
row-pair scorer and a dynamic last/first seam scorer:

- `serial_raster="scalable"`
- `serial_raster_wrap_strands=true`
- `serial_raster_identity_padding=true`
- `cyclic_band_generators=true`
- `max_strands=12`
- a fresh B* checkpoint and replay buffer; no ordinary checkpoint is relabelled
- semantic witnesses remain replay-verifiable and compile to ordinary Artin
  braid words for certification

Training starts on a registered mixture of simple knots and representations at
every strand count 2 through 12, with explicit counts for 6 through 12. Its
first comparison is paired against the strongest fresh ordinary 12-strand
candidate on a held-out, identity-disjoint Q20 panel using the same seeds,
SIM64, EV4, L10/L1000 objectives, and action horizon 128.

Launch is deferred until the currently active fresh Q20-1 program releases its
cores. This avoids changing or slowing the main Q20 gate and keeps the B*
comparison additive. Promotion requires retention at least 0.80 for every
strand count 6 through 12, no capacity exceptions, and a higher success-per-CPU
rate or strictly better paired coverage/capped L1000 than the ordinary system.
