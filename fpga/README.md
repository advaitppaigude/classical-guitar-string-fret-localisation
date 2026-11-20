# FPGA extension workspace

This directory is prepared for independent work described in the
[project specification](../guitar_fpga_project_specification.md). It currently
contains directory scaffolding only. The original Python software remains in
[`src/`](../src/); see the [software guide](../docs/software_maintenance.md) for
setup, tests, and reusable training/inference functions.

| Directory | Intended contents |
| --- | --- |
| `python/` | Future Python extension code |
| `models/` | Future model artifacts |
| `tests/` | Future extension software tests |
| `rtl/` | Future RTL sources |
| `tb/` | Future simulation testbenches |
| `quartus/` | Future board project files |
| `host/` | Future host software |
| `results/tables/` | Future result tables |
| `results/plots/` | Future result plots |
| `docs/` | Future extension documentation |

No model export, fixed-point algorithms, numerical experiments, RTL,
verification, board integration, or host communication is implemented here.
