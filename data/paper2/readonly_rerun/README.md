# Historical read-only rerun logs

This directory preserves raw read-only rerun logs and Cmt provenance from an
earlier audit. Any `Foot_error` line in the raw stdout is deprecated historical
reporter output and is not the horizontal foot-placement metric used by the
manuscript.

The authoritative landing-metric records, formula, validation report, and
controller summaries are in `supplementary/S4/landing_metric/`. Current
manuscript-facing tables attach those S4 records and do not read a landing value
from this historical directory. The raw logs remain unchanged so that earlier
Cmt, execution-time, source-hash, and status evidence can be audited.
