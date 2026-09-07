# From Git Clone to Running the Demo

This guide reproduces the validated public Unitree G1 / Isaac Lab environment.
The default directory layout is:

~~~text
workspace/
├── g1pickandplace-public/
├── unitree_sim_isaaclab/
├── unitree_ros/
├── unitree_sdk2_python/
├── IsaacLab/
└── cyclonedds/
~~~

The setup script creates the five sibling dependency directories listed after
`g1pickandplace-public`. Do not place private projects or assets in those paths.

## 1. Prerequisites

- Ubuntu 22.04 or newer. The current validation host uses Ubuntu with an
  RTX 5090 GPU.
- An NVIDIA GPU and driver that meet the Isaac Sim requirements.
- Git, Conda, and an account that can run `sudo apt-get`.
- Network access to GitHub, the NVIDIA Python index, the PyTorch wheel index,
  and Unitree's Hugging Face asset repository.
- Sufficient disk space. Isaac Sim, Isaac Lab, and the simulation assets are
  substantially larger than this source repository.

The script invokes Unitree's official `auto_setup_env.sh`. It therefore
installs system packages, downloads large assets, and may ask you to accept the
NVIDIA EULA or generate local certificates. These are first-time installation
operations; do not run the script over an existing production Conda
environment.

## 2. Clone and Install

~~~bash
mkdir -p workspace
cd workspace
git clone https://github.com/uhdfhnn/g1pickandplace-public.git
cd g1pickandplace-public
bash scripts/setup_environment.sh
~~~

The default Conda environment name is `unitree_sim_env`. If that name is
already in use on your machine, select a new environment name:

~~~bash
G1PICKPLACE_CONDA_ENV=g1_demo_env bash scripts/setup_environment.sh
~~~

The script stops if it finds a Conda environment with the selected name or a
modified dependency repository. It never deletes or overwrites those existing
resources automatically.

## 3. Pinned Versions

| Dependency | Validated version or commit | Basis and upgrade risk |
| --- | --- | --- |
| Isaac Sim | 5.0.0 | Unitree's recommended path for RTX 50-series GPUs. Version 4.5 may lack the required GPU support; 5.1 has not passed this project's visible gates. |
| Python | 3.11 | Version installed by Unitree's Isaac Sim 5.0 setup. Other versions may not have compatible wheels. |
| PyTorch | 2.7.0 | Validated with Isaac Sim 5.0 and Isaac Lab v2.2.0. NVIDIA may resolve the local CUDA suffix to a compatible build. |
| Isaac Lab | `46dff135f44683f031edf346e544fcfd8456b2bb` (`v2.2.0`) | Task and API version used for validation. An upgrade may change the scene, action terms, or sensor APIs. |
| `unitree_sim_isaaclab` | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Public scene, task registration, and asset layout used for validation. |
| `unitree_ros` | `7d6075f7f58588b189b940130e3edab3c839b2df` | Provides the validated G1 29-DoF URDF and meshes. |
| `unitree_sdk2_python` | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` | DDS Python interface used by Unitree's official installation. |
| CycloneDDS | `5041f3560c088c99e5088b2b8520b69169621196` | Validated 0.10.x build for the SDK. An upgrade may change DDS ABI or initialization behavior. |
| teleimager | `b81de448bca9c696d7ce145f4af71c66146d0b69` | Camera submodule pinned by `unitree_sim_isaaclab`. |
| Pinocchio (`pin`) | 2.7.0 | Validated IK and cmeel/HPP-FCL ABI. Version 3.x has not been validated on the current trajectories. |
| LeRobot | 0.4.4 (dataset-only) | Native v3 write, finalize, reopen, and video-validation API used by the dataset pipeline. Unrelated policy and training dependencies are not installed. |

These pins are a reproducibility lock, not a claim that every machine must use
only these versions. After changing any item, rerun the dependency check, full
unit test suite, visible inspect and plan gates, physical rollout, and LeRobot
reopen validation.

## 4. Verify the Installation

~~~bash
conda activate unitree_sim_env
cd workspace/g1pickandplace-public
python scripts/check_install.py
python -m pytest -q
python -m compileall -q src scripts tests
git diff --check
~~~

If you selected a custom environment name, replace the first command with that
name. `check_install.py` verifies the core modules, validated software
versions, and default sibling Unitree repositories.

LeRobot uses the exact dataset-only dependencies in
[`requirements-recording.txt`](requirements-recording.txt), installed with
`--no-deps`. Resolving LeRobot's complete training dependency set would replace
packages supplied by Isaac Sim; this project uses only the `LeRobotDataset`
write, finalize, reopen, and video-validation interfaces. At the end of setup,
the script imports that class for real. A missing dependency therefore fails
installation instead of leaving an environment that appears successful but
cannot record. This environment does not claim to support LeRobot policy
training.

## 5. Assimp / HPP-FCL Compatibility

On the validation host, Pinocchio 2.7.0's cmeel dependencies require
`cmeel.prefix/lib/libassimp.so.5` to be preloaded. Without it, HPP-FCL may
report symbol errors during startup. `scripts/run_demo.py` searches these
locations in order:

1. The explicit `--assimp-preload` path.
2. The `G1PICKPLACE_ASSIMP_LIB` environment variable.
3. The active `CONDA_PREFIX`.
4. `envs/<environment-name>` below the Conda installation directory.

Automatic discovery does not guess a system ABI. If the symbol error occurs
on your machine, provide the library explicitly:

~~~bash
export G1PICKPLACE_ASSIMP_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/cmeel.prefix/lib/libassimp.so.5"
test -f "$G1PICKPLACE_ASSIMP_LIB"
~~~

The Python 3.11 path follows the pinned environment above. If you change the
Python or cmeel version, locate the actual library and revalidate it instead of
copying this path unchanged.

## 6. Run the Safety Gates

Activate the environment and run from the repository root. By default, the
wrapper runs only visible inspect and plan; it does not execute a physical
rollout:

~~~bash
conda activate unitree_sim_env
cd workspace/g1pickandplace-public
python scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block."
~~~

Request rollout and native LeRobot recording only after both inspect and plan
pass:

~~~bash
python scripts/run_demo.py \
  --instruction "Pick up the red block and stack it on the yellow block." \
  --rollout
~~~

See [`docs/RUN_ENTRANCE_TEST_DEMO.md`](docs/RUN_ENTRANCE_TEST_DEMO.md) for the
detailed gate commands and acceptance criteria.

## 7. Keep Generated Evidence Out of Source Git

`outputs/`, `datasets/`, `videos/`, and `deliverables/` contain generated
artifacts and are ignored. The existing evaluator archive exceeds GitHub's
normal per-file limit. To distribute it, verify the manifest and SHA-256 first,
then use a GitHub Release asset, object storage, or explicitly configured Git
LFS. Do not mix generated evidence into ordinary source commits.
