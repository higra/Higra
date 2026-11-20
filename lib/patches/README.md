# Vendored dependency notes

pybind11 is installed by `lib/update_lib.sh` without local source patches, from
release v3.1.0 (commit `97bf890db679505a14dfe547a5e77bb2bd05dc90`).
The source archive used for this refresh has SHA-256
`ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665`.
The script can be run from any directory: it works in `lib/`, refuses to run
if `lib/` does not contain `FindTBB.cmake` and `patches/README.md`, and
builds and installs pybind11 into `lib/tmp/stage/` before replacing the previous
installation, which it restores if the replacement fails.

No local patches are currently applied to vendored libraries.
