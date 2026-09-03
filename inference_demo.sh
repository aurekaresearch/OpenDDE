

# ==============================================================================
# OpenDDE Model Inference Test Script
#
# Purpose:
#   This script provides usage examples for running inference with various
#   OpenDDE model versions and configurations.
#
# Arguments Summary (for 'opendde pred' or 'runner/inference.py'):
#   -i, --input (str):       [Required] Input JSON file or directory.
#   -o, --out_dir (str):     [Default: ./output] Output directory for results.
#   -s, --seeds (str):       [Default: 101] Inference seeds (e.g., "101,102").
#   -c, --cycle (int):       [Default: 10] Number of Pairformer cycles.
#   -p, --step (int):        [Default: 200] Number of diffusion steps.
#   -e, --sample (int):      [Default: 5] Samples per seed.
#   -d, --dtype (str):       [Default: bf16] Inference data type (bf16, fp32).
#   -n, --model_name (str):  [Default: opendde_v1] Model name.
#                            NOTE: opendde_v1 is the RECOMMENDED default.
#   --use_msa (bool):        Whether to use protein MSA features.
#   --use_default_params:    Auto-load recommended defaults for the model.
#   --trimul_kernel (str):   Triangle multiplicative kernel ('cuequivariance', 'torch').
#   --triatt_kernel (str):   Triangle attention kernel ('cuequivariance' or 'torch').
#   --use_template (bool):   Enable template features (v1.0.0+ only).
#   --use_rna_msa (bool):    Enable RNA MSA features (v1.0.0+ only).
#   --use_seeds_in_json:     Prioritize seeds defined in the input JSON.
#
# Available Models (Ref: opendde/config/model_registry.py, docs/supported_models.md):
#   * opendde_v1: [DEFAULT] Model supporting Template & RNA MSA with 96-bin distogram head.
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ------------------------------------------------------------------------------
# Section 1: Running via OpenDDE CLI (opendde pred)
# ------------------------------------------------------------------------------

# ##############################################################################
# # !!! IMPORTANT: ENVIRONMENT SETUP !!!
# # ----------------------------------------------------------------------------
# # 1. Ensure environment variables are correctly set:
# #    - OPENDDE_ROOT_DIR: Your data root directory
# #
# #    Uncomment and modify the line below if needed:
# #    # export OPENDDE_ROOT_DIR="/modify/to/your/data_root_dir"
# #
# # 2. Dependency for Template & RNA MSA:
# #    If using these features, ensure 'kalign' and 'hmmer' are installed:
# #    apt-get update && apt-get install -y kalign hmmer
# # ############################################################################

echo "Starting Section 1: CLI-based inference tests..."

# Example 1.1: Standard inference with Template support
opendde pred \
    -i examples/input.json \
    -o ./test_outputs/cmd/output_base_v1 \
    -s 101 \
    -n opendde_v1 \
    --use_template true \
    --use_default_params true


# Example 1.2: Inference using seeds defined in JSON
opendde pred \
    -i examples/examples_with_template/example_mgyp004658859411.json \
    -o ./test_outputs/cmd/output_base_v1 \
    -s 101 \
    -n opendde_v1 \
    --use_template true \
    --use_seeds_in_json true \
    --use_default_params true

# Example 1.3: RNA MSA support
opendde pred \
    -i examples/examples_with_rna_msa/example_9gmw_2.json \
    -o ./test_outputs/cmd/output_base_v1 \
    -n opendde_v1 \
    --use_rna_msa true  \
    --use_default_params true

# Example 1.4: Lightweight single-sample run
opendde pred \
    -i examples/input.json \
    -o ./test_outputs/cmd/output_base_v1_single \
    -s 101 \
    -n opendde_v1 \
    --use_template true \
    --use_default_params true


# ------------------------------------------------------------------------------
# Section 2: Running via Runner Script (runner/inference.py)
#
# IMPORTANT:
#   Direct script execution requires features (MSA, templates, RNA MSA, etc.)
#   to be pre-prepared in the input JSON. This mode is optimized for GPU-only
#   computation.
#   If features are NOT ready, please use the preprocessing command first:
#   Example: opendde prep --input examples/input.json --out_dir ./output
# ------------------------------------------------------------------------------

echo "Starting Section 2: Script-based inference tests..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Test 2.1: OpenDDE v1 with Template support
# Features: Template enabled, cuequivariance attention
N_sample=5
N_step=200
N_cycle=10
seed=103
input_json_path="./examples/examples_with_template/example_9fm7.json"
dump_dir="./test_outputs/sh/output_m_9fm7"
model_name="opendde_v1"

python3 runner/inference.py \
    --model_name ${model_name} \
    --seeds ${seed} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --triangle_attention "cuequivariance" \
    --use_seeds_in_json true \
    --triangle_multiplicative "cuequivariance" \
    --use_template true

# Test 2.2: OpenDDE v1 direct runner example
N_sample=1
N_step=200
N_cycle=10
seed=101
input_json_path="./examples/input.json"
dump_dir="./test_outputs/sh/output_base_u96"
model_name="opendde_v1"

python3 runner/inference.py \
    --model_name ${model_name} \
    --seeds ${seed} \
    --dump_dir ${dump_dir} \
    --input_json_path ${input_json_path} \
    --model.N_cycle ${N_cycle} \
    --sample_diffusion.N_sample ${N_sample} \
    --sample_diffusion.N_step ${N_step} \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --use_template true
echo "All inference tests completed."
