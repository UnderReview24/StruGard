# SGCR 10k / four domains / memory 500

This is the next requested experiment configuration, saved on 2026-09-08.
Only settings have been prepared. No dataset conversion, model training,
preflight optimizer updates, or experiment queue is launched by this change.

## Protocol

- Config: `configs/experiment/sgcr_10k_4d_m500.yaml`.
- Total selected dataset: 10,000 structures, seed 42, using the existing
  deterministic source-row permutation. This is total data, not 10,000
  training records. The selected IDs will contain the old 5,000 IDs.
- Four crystal-system groups, with no target-based grouping or per-domain cap:
  triclinic/monoclinic; orthorhombic/tetragonal; hexagonal/trigonal; cubic.
- The existing builder sorts group names into domain IDs. Seed 42 gives
  presentation order `[3, 2, 1, 0]`, the group order listed above.
- Keep the existing per-domain 80/10/10 train/validation/test rule. Counts
  are rounded per domain; exact counts and minimum-domain-size validation
  remain pending until the 10,000-sample data are prepared. Domains are not
  forced to have equal sizes.
- Total retained historical memory is at most 500 across all domains.
  The existing 80/20 partition stays inside this budget: at full capacity,
  400 examples are eligible for gradient replay and 100 are reserved.
  This is not 500 per domain or 500 per minibatch.
- Keep batch size 32, current-to-replay ratio 1:1, 10 epochs per post-D1
  stage, D1 training settings, seed, optimizer, architecture, and SGCR
  hyperparameters from the 5k preset. The 6,000 storage-candidate cap is
  temporary selection capacity, not additional retained replay memory.
- Use separate processed data, domain files, D1 checkpoint, feature caches,
  reference outputs, and run outputs. Graph files may share the existing
  content/configuration-keyed cache. All old 5k artifacts remain historical.

## Before a future run

The current SGCR launch/preflight/audit/report scripts are specialized to
the completed 5k benchmark and are not yet launchers for this new preset.
Do not replace their default config path and assume the rest is compatible.
Running `scripts/50_sgcr.py` unchanged still selects the old 5k config.

1. Prepare 10,000 selected records in the new processed directory; create
   and freeze the four-domain splits, then record their actual hashes.
   New split hashes are deliberately null rather than inherited from 5k.
2. Prepare the new static baseline/shift validation, D1-only checkpoint,
   and matching structural feature caches using the new data provenance.
3. Adapt the SGCR runner and checks from fixed 5,000 samples, three stages,
   810 optimizer steps, 503 test examples, and 2,000 memory entries to the
   new resolved config and actual split sizes. Add config propagation to
   launchers and use four-stage aggregation/reporting.
4. Establish a new Structural Replay control and Direct Predictor reference
   on this benchmark. Do not compare predictions or traces against the old
   5k baseline as a reproduction gate; `baseline_reference` is unset.

Increasing sample size, changing domain count, and reducing memory all change
the benchmark. A future comparison must train its controls on the same new
protocol; comparing final MAE directly with 5k does not isolate sample size.
