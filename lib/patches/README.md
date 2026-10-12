# Local vendor patches

`xtensor-python-pybind11-3.patch` extends the compound-assignment operator guards
in xtensor-python 0.28.0's `pyarray.hpp` and `pytensor.hpp` to pybind11 3.x. Without
the patch, both containers inherit ambiguous compound-assignment operators from
xtensor and pybind11. `lib/update_lib.sh` applies it after installation; run that
script from `lib/` as usual, never from the repository root.

The guard matches upstream commit
[afc1672](https://github.com/xtensor-stack/xtensor-python/commit/afc1672f50e2901f95a32b4cd9f480fa28545468)
for `pyarray.hpp` and applies the same condition to `pytensor.hpp`. The numerical dependency
versions remain unchanged. The optional embedded test target exercises NumPy
input/output and in-place compound assignment for both containers.

pybind11 itself is installed without local source patches from release v3.1.0
(commit `97bf890db679505a14dfe547a5e77bb2bd05dc90`).
The source archive used for this refresh has SHA-256
`ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665`.
