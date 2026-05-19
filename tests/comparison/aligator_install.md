# Installing Aligator for the comparison driver

Aligator (https://github.com/Simple-Robotics/aligator) is a Boost.Python /
CMake project. Its PyPI listing (`Aligator 1.1.0` on
pypi.org) is **a different package** —
the real aligator is conda-forge only. Neither `pip install aligator`
nor `uv add aligator` will work.

This document records what worked and how the comparison adapter
(`tests/comparison/adapters/aligator.py`) finds the conda install from
the project's uv venv.

## Install path: conda-forge into a side env

```sh
mamba create -n aligator-side -c conda-forge -y python=3.13 aligator
```

This produces a Python 3.13 environment at
`~/.conda/envs/aligator-side/` containing aligator plus its runtime
dependencies (proxsuite-nlp, eigenpy, pinocchio, hpp-fcl/coal,
boost-python, libstdc++, etc.) under
`~/.conda/envs/aligator-side/lib/`. Check the install with:

```sh
~/.conda/envs/aligator-side/bin/python -c "import aligator; print(aligator.__version__)"
```

The adapter targets the public surface (`SolverProxDDP`,
`ExplicitDynamicsModel`, `CostAbstract`, `StageFunction`,
`constraints.NegativeOrthant`, `constraints.EqualityConstraintSet`)
that is stable across the recent conda-forge builds.

## How the uv venv finds it

The project's uv venv (`/home/joapinto/github/primal-dual-lipa/.venv`)
is also Python 3.13. The aligator extension is a single
`.so`(`pyaligator.cpython-313-x86_64-linux-gnu.so`) inside its package
directory; its RPATH is `$ORIGIN/../../../`, so it finds its own
sibling shared libraries (`libaligator.so`, `libproxsuite-nlp.so`,
`libpinocchio_default.so`, `libboost_python313.so`, etc.) inside the
conda env's `lib/` regardless of the importing process's
`LD_LIBRARY_PATH`.

The adapter therefore just prepends the conda env's `site-packages` to
`sys.path` at import time and re-uses the host venv's numpy. Quick
sanity check:

```sh
UV_NO_BUILD_ISOLATION=0 uv run --group comparisons python -c \
  "import sys; sys.path.insert(0, '/home/joapinto/.conda/envs/aligator-side/lib/python3.13/site-packages'); \
   import aligator; print(aligator.__version__)"
```

The host venv's numpy 2.x is ABI-compatible with the conda numpy that
aligator was built against.

## Things that did not work

* `pip install aligator` (PyPI): there is a *different* package called
  `Aligator 1.1.0` on PyPI (https://pypi.org/project/aligator/) which
  is an empty stub (an 889-byte zip with no `setup.py` or
  `pyproject.toml`). It is **not** the Inria/LAAS aligator solver and
  cannot be installed.
* Cloning Simple-Robotics/aligator and running `pip install .`: the
  project is CMake-only with no Python build backend declared. The
  upstream supports `pixi` and Conda for development.
* Building with `pip wheel` from source needs the full transitive
  dependency stack (proxsuite-nlp, pinocchio, eigenpy, coal, plus
  Boost.Python 1.86) at compile time, all of which are themselves
  CMake projects. Reproducing that without conda is hours of work and
  brings no advantage over the conda install.

## Adapter discovery rule

`tests/comparison/adapters/aligator.py` looks up the side conda env in
this order:

1. The `ALIGATOR_SITE_PACKAGES` env var (if set, used directly).
2. `~/.conda/envs/aligator-side/lib/python3.13/site-packages` (default
   path matching the `mamba create` recipe above).
3. Whatever is on `sys.path` already — i.e. if you happen to run the
   benchmark *from* the conda env, no path manipulation is needed.

If none of those provides an importable `aligator`, the adapter
returns `unavailable` with a diagnostic pointing back at this file.

## Boost.Python coexistence with `csqp`

Aligator and `mim_solvers` (the `csqp` adapter's backend) both link
against pinocchio + eigenpy + Boost.Python. The conda-forge aligator
0.12.0 ships its own pinocchio 3.4 / eigenpy / proxsuite-nlp under
`~/.conda/envs/aligator-side/lib/`; the project venv has pinocchio 3.8
+ libcrocoddyl 3.2 + mim-solvers under `.venv/.../cmeel.prefix/`.
Boost.Python registers a global type-conversion table at first
import. Once one of the two ecosystems has been loaded into the
process, the other one's `_pywrap` extension fails with either
`AttributeError: 'NoneType' object has no attribute '__dict__'` or
`TypeError: No to_python (by-value) converter found for C++ type:
boost::none_t`.

The adapter handles this by:

* **Eagerly preloading aligator at module-import time**, before any
  competing import has a chance to run.
* Refusing to preload (and reporting unavailable) if `pinocchio`,
  `eigenpy`, `crocoddyl`, or `mim_solvers` is already in `sys.modules`
  by the time the adapter module loads. (In the standard adapter load
  order from `tests/comparison/adapters/__init__.py` this never
  happens, since every adapter does its solver imports lazily inside
  `is_available` / `solve`.)

Practical consequence: in a single process you can run **either**
`aligator + lipa + ipopt + acados` **or** `csqp + lipa + ipopt +
acados`, but not `aligator + csqp` together. The `run_benchmark` CLI
will report `csqp` as `unavailable: libpinocchio_parsers.so.3 cannot
open shared object file` when both are requested, because aligator's
preload moved the conda env to the front of `sys.path` and the conda
env's pinocchio is missing the parsers .so that `mim_solvers` needs.

If you need both in one report, run two passes and concatenate the
CSVs / markdowns.
