# Historical raised-L1145-r1 provenance record

This directory is immutable provenance for an earlier source-aligned run. Its
raw stdout and frozen learned-controller source snapshots preserve the reported
`Cmt`, execution time, recovery settings, and source hashes.

The `Foot_error` lines in the raw stdout and the corresponding reporter code in
the frozen snapshots are deprecated historical output. They are not the
horizontal foot-placement metric used by the manuscript. The current metric is
defined and recomputed in `supplementary/S4/landing_metric/`, and the current
Table V generator does not read a landing value from this directory.

The current learned-controller entrypoints use grid-selected recovery settings
and therefore do not reproduce the two historical learned-controller records.
The LIPM entrypoint remains source-aligned and is the only override consumed by
the current two-link table generator.
