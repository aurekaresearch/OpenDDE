# Docker Installation


Use Docker for GPU inference on a Linux host with an NVIDIA GPU. All examples
below are one-shot `docker run` commands executed from the host. For non-Docker
installation, see [inference_instructions.md](./inference_instructions.md).

## 1. Verify Docker GPU support

Install Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
then verify that containers can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## 2. Get the image

Pull the prebuilt image:

```bash
docker pull aurekaresearch/opendde:v1
```

The Docker `v1` tag is maintained separately from the Python package version.

Or build from the repository root:

```bash
bash scripts/build_docker_image.sh
```

The helper performs one local build tagged `opendde:local`, targeting
`linux/amd64` by default. It does not inspect Git state, attach repository
release metadata, push an image, or run from CI. Use `--tag`, `--platform`,
`--pull`, or `--no-cache` as needed; run it with `--help` for the complete
interface. If you use the locally built image, replace
`aurekaresearch/opendde:v1` with `opendde:local` in the examples below.

## 3. Prepare runtime data

OpenDDE reads checkpoints and runtime data from `OPENDDE_ROOT_DIR`:

```bash
export OPENDDE_ROOT_DIR="$PWD/opendde_data"
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
```

Released checkpoints keep the public filenames `opendde.pt` and
`opendde_abag.pt`. Their authoritative download links and digests are listed in
[supported_models.md](./supported_models.md).

Download or verify the released checkpoint and remaining runtime files with
Docker. The helper validates official checkpoints against the bundled manifest
before atomically installing them:

```bash
docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data
```

To download only the released ABAG checkpoint:

```bash
docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data \
    --skip-common \
    --skip-search-database \
    --checkpoint opendde_abag.pt
```

Select that released checkpoint explicitly for an ABAG run:

```bash
--load_checkpoint_path /opendde_data/checkpoint/opendde_abag.pt
```

If you already have a custom checkpoint, keep its own descriptive filename and
copy it into the mounted checkpoint directory. Prepare only the remaining
runtime files with `--skip-model`, so the helper neither validates the custom
file as a released asset nor installs the unrelated default checkpoint:

```bash
cp /absolute/path/to/my_checkpoint.pt \
  "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt"

docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data \
    --skip-model
```

Select the custom checkpoint explicitly during inference:

```bash
--load_checkpoint_path /opendde_data/checkpoint/my_checkpoint.pt
```

For protein-only smoke tests that disable MSA/template/RNA-MSA preprocessing, you
can skip search databases:

```bash
docker run --rm \
  -v "$OPENDDE_ROOT_DIR":/opendde_data \
  aurekaresearch/opendde:v1 \
  bash scripts/download_opendde_data.sh \
    --root /opendde_data \
    --skip-search-database
```

## 4. Run inference

The command below assumes `tiny.json` exists in the current host directory. See
[../README.md](../README.md) for the minimal input example.

```bash
mkdir -p output

docker run --rm --gpus all --shm-size=4g \
  -e OPENDDE_ROOT_DIR=/opendde_data \
  -v "$OPENDDE_ROOT_DIR":/opendde_data:ro \
  -v "$PWD":/workspace \
  -v "$PWD/output":/output \
  aurekaresearch/opendde:v1 \
  opendde pred \
    -i /workspace/tiny.json \
    -o /output \
    -n opendde_v1 \
    --use_msa false \
    --use_template false \
    --use_rna_msa false \
    --sample 1 \
    --step 200 \
    --cycle 10
```

For production inference options, MSA/template preprocessing, and checkpoint
configuration, see [inference_instructions.md](./inference_instructions.md).
