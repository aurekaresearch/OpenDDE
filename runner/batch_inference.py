# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import difflib
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional, Union

import click
import tqdm
from Bio import SeqIO
from rdkit import Chem

from opendde.config.inference import (
    build_inference_config,
    update_gpu_compatible_configs,
)
from opendde.config.model_registry import DEFAULT_MODEL_NAME, model_configs
from opendde.data.inference.json_maker import cif_to_input_json
from opendde.data.inference.json_parser import lig_file_to_atom_info
from opendde.data.utils import pdb_to_cif
from opendde.distributed.foldcp.config import FoldCPConfig, apply_foldcp_config
from opendde.utils.download import download_inference_cache
from opendde.utils.environment import format_doctor_report
from opendde.utils.logger import get_logger
from opendde.utils.logging_config import init_logging
from opendde.version import __version__
from runner.inference import (
    InferenceRunner,
    infer_predict,
)
from runner.msa_search import msa_search, update_infer_json
from runner.rna_msa_search import update_rna_msa_info
from runner.template_search import update_template_info

logger = get_logger(__name__)
SUPPORTED_MODELS = tuple(model_configs.keys())


def preprocess_input(
    input_json: str,
    out_dir: str,
    use_msa: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
    msa_server_mode: Optional[str] = None,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
) -> str:
    """
    Preprocess the input JSON file by performing MSA, template, and RNA MSA searches as needed.

    Args:
        input_json (str): Path to the input JSON file.
        out_dir (str): Directory to save search results.
        use_msa (bool): Whether to use protein MSA.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        msa_server_mode (Optional[str]): Deprecated compatibility argument; ignored.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.

    Returns:
        str: Path to the updated JSON file.
    """
    # 1. Protein MSA search
    msa_updated_json, _ = update_infer_json(
        input_json, out_dir, use_msa=use_msa, mode=msa_server_mode
    )

    # Read the data (either original or updated)
    with open(msa_updated_json, "r") as f:
        json_data = json.load(f)

    actual_updated = False

    # 2. Template search
    if use_template:
        template_updated = update_template_info(
            json_data,
            hmmsearch_binary_path=hmmsearch_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            seqres_database_path=seqres_database_path,
        )
        actual_updated = actual_updated or template_updated

    # 3. RNA MSA search
    if use_rna_msa:
        rna_updated = update_rna_msa_info(
            json_data,
            out_dir=out_dir,
            nhmmer_binary_path=nhmmer_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            hmmbuild_binary_path=hmmbuild_rna_binary_path or hmmbuild_binary_path,
            ntrna_database_path=ntrna_database_path,
            rfam_database_path=rfam_database_path,
            rna_central_database_path=rna_central_database_path,
            nhmmer_n_cpu=nhmmer_n_cpu,
        )
        actual_updated = actual_updated or rna_updated

    if actual_updated:
        base, ext = os.path.splitext(os.path.basename(msa_updated_json))
        if "-update-msa" in base:
            output_json_name = base.replace("-update-msa", "-final-updated") + ext
        else:
            output_json_name = f"{base}-final-updated{ext}"

        output_json = os.path.join(
            os.path.dirname(os.path.abspath(msa_updated_json)), output_json_name
        )

        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=4)
        logger.info(f"Input preprocessing completed, results saved to {output_json}")
        return output_json
    else:
        return msa_updated_json


def generate_infer_jsons(protein_msa_res: dict, ligand_file: str) -> List[str]:
    """
    Generate inference JSON files from protein MSA results and ligand files.

    Args:
        protein_msa_res (dict): Dictionary mapping protein sequences to their MSA results.
        ligand_file (str): Path to a ligand file (SDF or SMI) or directory containing ligand files.

    Returns:
        List[str]: List of paths to the generated inference JSON files.
    """
    protein_chains = []
    if len(protein_msa_res) <= 0:
        raise RuntimeError(f"invalid `protein_msa_res` data in {protein_msa_res}")
    for key, value in protein_msa_res.items():
        protein_chain = {}
        protein_chain["proteinChain"] = {}
        protein_chain["proteinChain"]["sequence"] = key
        protein_chain["proteinChain"]["count"] = value.get("count", 1)
        protein_chain["proteinChain"]["msa"] = value
        protein_chains.append(protein_chain)
    if os.path.isdir(ligand_file):
        ligand_files = [
            str(file) for file in Path(ligand_file).rglob("*") if file.is_file()
        ]
        if len(ligand_files) == 0:
            raise RuntimeError(
                f"can not read a valid `sdf` or `smi` ligand_file in {ligand_file}"
            )
    elif os.path.isfile(ligand_file):
        ligand_files = [ligand_file]
    else:
        raise RuntimeError(f"can not read a special ligand_file: {ligand_file}")

    invalid_ligand_files = []
    sdf_ligand_files = []
    smi_ligand_files = []
    tmp_json_name = uuid.uuid4().hex
    current_local_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}"
    )
    current_local_json_dir = (
        f"/tmp/{time.strftime('%Y-%m-%d', time.localtime())}/{tmp_json_name}_jsons"
    )
    os.makedirs(current_local_dir, exist_ok=True)
    os.makedirs(current_local_json_dir, exist_ok=True)
    for li_file in ligand_files:
        try:
            if li_file.endswith(".smi"):
                smi_ligand_files.append(li_file)
            elif li_file.endswith(".sdf"):
                suppl = Chem.SDMolSupplier(li_file)
                if len(suppl) <= 1:
                    lig_file_to_atom_info(li_file)
                    sdf_ligand_files.append([li_file])
                else:
                    sdf_basename = os.path.join(
                        current_local_dir, os.path.basename(li_file).split(".")[0]
                    )
                    li_files = []
                    for idx, mol in enumerate(suppl):
                        p_sdf_path = f"{sdf_basename}_part_{idx}.sdf"
                        writer = Chem.SDWriter(p_sdf_path)
                        writer.write(mol)
                        writer.close()
                        li_files.append(p_sdf_path)
                        lig_file_to_atom_info(p_sdf_path)
                    sdf_ligand_files.append(li_files)
            else:
                lig_file_to_atom_info(li_file)
                sdf_ligand_files.append([li_file])
        except Exception as exc:
            logging.info(f" lig_file_to_atom_info failed with error info: {exc}")
            invalid_ligand_files.append(li_file)
    logger.info(f"the json to infer will be save to {current_local_json_dir}")
    infer_json_files = []
    for li_files in sdf_ligand_files:
        one_infer_seq = protein_chains[:]
        for li_file in li_files:
            ligand_name = os.path.basename(li_file).split(".")[0]
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = f"FILE_{li_file}"
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_sdf_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)

    for smi_ligand_file in smi_ligand_files:
        with open(smi_ligand_file, "r") as f:
            smile_list = f.readlines()
        one_infer_seq = protein_chains[:]
        ligand_name = os.path.basename(smi_ligand_file).split(".")[0]
        for smile in smile_list:
            normalize_smile = smile.replace("\n", "")
            ligand_chain = {}
            ligand_chain["ligand"] = {}
            ligand_chain["ligand"]["ligand"] = normalize_smile
            ligand_chain["ligand"]["count"] = 1
            one_infer_seq.append(ligand_chain)
        one_infer_json = [{"sequences": one_infer_seq, "name": ligand_name}]
        json_file_name = os.path.join(
            current_local_json_dir, f"{ligand_name}_smi_{uuid.uuid4().hex}.json"
        )
        with open(json_file_name, "w") as f:
            json.dump(one_infer_json, f, indent=4)
        infer_json_files.append(json_file_name)
    if len(invalid_ligand_files) > 0:
        logger.warning(
            f"Found {len(invalid_ligand_files)} invalid ligand files. "
            f"Example: {invalid_ligand_files[0]}"
        )
    return infer_json_files


def get_default_runner(
    seeds: Optional[list[int]] = None,
    dump_dir: str = "./output",
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: Literal["bf16", "fp32"] = "fp32",
    model_name: str = DEFAULT_MODEL_NAME,
    load_checkpoint_path: str = "",
    use_msa: bool = True,
    trimul_kernel="auto",
    triatt_kernel="auto",
    enable_cache=True,
    enable_fusion=True,
    enable_tf32=True,
    deterministic: bool = False,
    use_template: bool = False,
    use_rna_msa: bool = False,
    need_atom_confidence: bool = False,
    kalign_binary_path: Optional[str] = None,
    prot_template_mmcif_dir: Optional[str] = None,
    prot_template_cache_dir: Optional[str] = None,
    use_tfg_guidance: bool = False,
    foldcp_mode: Literal["single", "distributed"] = "single",
    foldcp_size_dp: int = 1,
    foldcp_size_cp: int = 1,
    foldcp_devices: str = "",
    foldcp_metrics_jsonl: str = "",
) -> InferenceRunner:
    """
    Get a default InferenceRunner with the specified configurations.

    Args:
        seeds (Optional[list]): List of inference seeds.
        dump_dir (str): Output directory for results.
        n_cycle (int): Number of Pairformer cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Inference data type. Defaults to 'fp32'.
        model_name (str): Name of the model checkpoint.
        load_checkpoint_path (str): Explicit checkpoint path. If unset, uses the released checkpoint filename for model_name.
        use_msa (bool): Whether to use MSA.
        trimul_kernel (str): Kernel for triangle multiplicative update.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Whether to enable diffusion shared variables cache.
        enable_fusion (bool): Whether to enable diffusion transformer fusion.
        enable_tf32 (bool): Whether to enable TF32.
        deterministic (bool): Whether to enable deterministic PyTorch algorithms.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        use_tfg_guidance (bool): Whether to use TFG guidance.
        foldcp_mode (str): Fold-CP execution mode.
        foldcp_size_dp (int): Number of data-parallel ranks.
        foldcp_size_cp (int): Number of context-parallel ranks.
        foldcp_devices (str): Optional visible device list recorded in metrics.
        foldcp_metrics_jsonl (str): Optional JSONL path for benchmark records.

    Returns:
        InferenceRunner: An instance of InferenceRunner.
    """
    foldcp_config = FoldCPConfig.from_runtime_args(
        mode=foldcp_mode,
        size_dp=foldcp_size_dp,
        size_cp=foldcp_size_cp,
        devices=foldcp_devices,
        metrics_jsonl=foldcp_metrics_jsonl,
    )
    # Merge model-specific overrides into the raw config BEFORE parsing, so that
    # GlobalConfigValue references (e.g. submodule c_z) resolve against the
    # model's top-level values. This mirrors runner.inference.run(); doing the
    # update AFTER parse_configs would leave submodules at base defaults
    # (e.g. confidence_head c_z=128) and break strict checkpoint loading.
    configs = build_inference_config(
        model_name=model_name,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    model_name = configs.model_name
    # the user input configs has the highest priority
    configs.dump_dir = dump_dir
    configs.load_checkpoint_path = load_checkpoint_path
    configs.model.N_cycle = n_cycle
    configs.sample_diffusion.N_sample = n_sample
    configs.sample_diffusion.N_step = n_step
    configs.dtype = dtype
    configs.use_msa = use_msa
    configs.triangle_multiplicative = trimul_kernel
    configs.triangle_attention = triatt_kernel
    configs.enable_diffusion_shared_vars_cache = enable_cache
    configs.enable_efficient_fusion = enable_fusion
    configs.enable_tf32 = enable_tf32
    configs.deterministic = deterministic
    configs.use_template = use_template
    configs.use_rna_msa = use_rna_msa
    configs.need_atom_confidence = need_atom_confidence
    configs.sample_diffusion.guidance["enable"] = use_tfg_guidance
    configs = apply_foldcp_config(configs, foldcp_config)

    if kalign_binary_path is not None:
        # The path provided by the user is expected to exist by default
        configs.data.template.kalign_binary_path = kalign_binary_path
        assert os.path.exists(kalign_binary_path), (
            f"kalign_binary_path {kalign_binary_path} does not exist"
        )
    else:
        # If no path is provided and templates are used, try to find kalign in the system PATH
        if use_template:
            found_path = None
            try:
                result = subprocess.run(
                    ["which", "kalign"], capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    kalign_in_path = result.stdout.strip()
                    if os.path.exists(kalign_in_path) and os.access(
                        kalign_in_path, os.X_OK
                    ):
                        found_path = kalign_in_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            if found_path is not None:
                configs.data.template.kalign_binary_path = found_path
            else:
                raise RuntimeError(
                    "Kalign binary not found in system PATH. "
                    "To install kalign, you can use one of the following methods:\n"
                    "1. Using conda: conda install -c bioconda kalign\n"
                    "2. Using apt (Ubuntu/Debian): apt-get install kalign\n"
                    "3. Download from: https://github.com/TimoLassmann/kalign\n"
                    "After installation, make sure the binary is accessible in PATH or provide kalign_binary_path."
                )

    if prot_template_mmcif_dir is not None:
        configs.data.template.prot_template_mmcif_dir = prot_template_mmcif_dir
    if prot_template_cache_dir is not None:
        configs.data.template.prot_template_cache_dir = prot_template_cache_dir

    configs = update_gpu_compatible_configs(configs)
    logger.info(
        f"Inference by OpenDDE: model_name: {model_name}, dtype: {configs.dtype}"
    )
    logger.info(
        f"Triangle_multiplicative kernel: {configs.triangle_multiplicative}, "
        f"Triangle_attention kernel: {configs.triangle_attention}"
    )
    logger.info(
        f"enable_diffusion_shared_vars_cache: {configs.enable_diffusion_shared_vars_cache}, "
        f"enable_efficient_fusion: {configs.enable_efficient_fusion}, "
        f"enable_tf32: {configs.enable_tf32}"
    )
    logger.info(
        "Fold-CP mode: %s, size_dp=%s, size_cp=%s, mesh=%s, metrics_jsonl=%s",
        foldcp_config.mode,
        foldcp_config.size_dp,
        foldcp_config.size_cp,
        foldcp_config.cp_mesh_shape,
        foldcp_config.metrics_jsonl or "<disabled>",
    )
    download_inference_cache(configs)
    return InferenceRunner(configs)


def inference_jsons(
    json_file: str,
    out_dir: str = "./output",
    use_msa: bool = True,
    seeds: Optional[list[int]] = None,
    n_cycle: int = 10,
    n_step: int = 200,
    n_sample: int = 5,
    dtype: Literal["bf16", "fp32"] = "fp32",
    model_name: str = DEFAULT_MODEL_NAME,
    load_checkpoint_path: str = "",
    trimul_kernel: str = "auto",
    triatt_kernel: str = "auto",
    enable_cache: bool = True,
    enable_fusion: bool = True,
    enable_tf32: bool = True,
    deterministic: bool = False,
    use_template: bool = False,
    use_rna_msa: bool = False,
    msa_server_mode: Optional[str] = None,
    need_atom_confidence: bool = False,
    kalign_binary_path: Optional[str] = None,
    prot_template_mmcif_dir: Optional[str] = None,
    prot_template_cache_dir: Optional[str] = None,
    use_tfg_guidance: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    foldcp_mode: Literal["single", "distributed"] = "single",
    foldcp_size_dp: int = 1,
    foldcp_size_cp: int = 1,
    foldcp_devices: str = "",
    foldcp_metrics_jsonl: str = "",
) -> None:
    """
    Run inference on a single JSON file or a directory of JSON files.

    Args:
        json_file (str): Path to a JSON file or directory containing JSON files.
        out_dir (str): Directory to save inference results.
        use_msa (bool): Whether to use MSA.
        seeds (Optional[list[int]]): List of inference seeds.
        n_cycle (int): Number of cycles.
        n_step (int): Number of diffusion steps.
        n_sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Model name.
        load_checkpoint_path (str): Explicit checkpoint path.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Enable shared variables cache.
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        deterministic (bool): Enable deterministic PyTorch algorithms.
        use_template (bool): Whether to use templates.
        use_rna_msa (bool): Whether to use RNA MSA.
        msa_server_mode (Optional[str]): Deprecated compatibility argument; ignored.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        prot_template_mmcif_dir (Optional[str]): Override for the protein template
            mmCIF directory; use a writable path when the data root is read-only.
        prot_template_cache_dir (Optional[str]): Override for the parsed template
            cache directory; use a writable path when the data root is read-only.
        use_tfg_guidance (bool): Use TFG guidance.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        foldcp_mode (str): Fold-CP execution mode.
        foldcp_size_dp (int): Number of data-parallel ranks.
        foldcp_size_cp (int): Number of context-parallel ranks.
        foldcp_devices (str): Optional visible device list recorded in metrics.
        foldcp_metrics_jsonl (str): Optional JSONL path for benchmark records.
    """
    infer_jsons = []
    if os.path.isdir(json_file):
        infer_jsons = [
            str(file) for file in Path(json_file).rglob("*") if file.is_file()
        ]
        if len(infer_jsons) == 0:
            raise RuntimeError(f"Can not read a valid json file in {json_file}")
    elif os.path.isfile(json_file):
        infer_jsons = [json_file]
    else:
        raise RuntimeError(f"Can not read a special file: {json_file}")
    infer_jsons = [file for file in infer_jsons if file.endswith(".json")]
    logger.info(f"Will infer with {len(infer_jsons)} jsons")
    if len(infer_jsons) == 0:
        return

    infer_errors = {}
    runner = get_default_runner(
        seeds=seeds,
        dump_dir=out_dir,
        n_cycle=n_cycle,
        n_step=n_step,
        n_sample=n_sample,
        dtype=dtype,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        use_msa=use_msa,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        enable_cache=enable_cache,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        deterministic=deterministic,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        need_atom_confidence=need_atom_confidence,
        kalign_binary_path=kalign_binary_path,
        prot_template_mmcif_dir=prot_template_mmcif_dir,
        prot_template_cache_dir=prot_template_cache_dir,
        use_tfg_guidance=use_tfg_guidance,
        foldcp_mode=foldcp_mode,
        foldcp_size_dp=foldcp_size_dp,
        foldcp_size_cp=foldcp_size_cp,
        foldcp_devices=foldcp_devices,
        foldcp_metrics_jsonl=foldcp_metrics_jsonl,

    )
    configs = runner.configs
    for _, infer_json in enumerate(tqdm.tqdm(infer_jsons)):
        try:
            configs["input_json_path"] = preprocess_input(
                infer_json,
                out_dir=out_dir,
                use_msa=use_msa,
                use_template=use_template,
                use_rna_msa=use_rna_msa,
                msa_server_mode=msa_server_mode,
                hmmsearch_binary_path=hmmsearch_binary_path,
                hmmbuild_binary_path=hmmbuild_binary_path,
                seqres_database_path=seqres_database_path,
                nhmmer_binary_path=nhmmer_binary_path,
                hmmalign_binary_path=hmmalign_binary_path,
                hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
                ntrna_database_path=ntrna_database_path,
                rfam_database_path=rfam_database_path,
                rna_central_database_path=rna_central_database_path,
                nhmmer_n_cpu=nhmmer_n_cpu,
            )
            infer_predict(runner, configs)
        except Exception as exc:
            infer_errors[infer_json] = str(exc)
    if len(infer_errors) > 0:
        logger.warning(f"Run inference failed: {infer_errors}")


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


class SuggestGroup(click.Group):
    """A Click group that suggests similar commands on error."""

    def resolve_command(self, ctx, args):
        """Try to resolve the command, and suggest matches if it fails."""
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            if len(args) > 0:
                cmd_name = args[0]
                all_commands = self.list_commands(ctx)
                matches = difflib.get_close_matches(cmd_name, all_commands)
                if matches:
                    e.message += (
                        f"\n\nDid you mean one of these?\n    {', '.join(matches)}"
                    )
            raise e


@click.group(name="opendde", cls=SuggestGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def opendde_cli() -> None:
    """
    OpenDDE: an AlphaFold 3-style structure prediction toolkit.

    This CLI provides tools for structure prediction, data conversion,
    and MSA/template searching.
    """
    pass


@click.command(context_settings=CONTEXT_SETTINGS)
def doctor() -> None:
    """Print environment diagnostics and install recommendations."""
    click.echo(format_doctor_report())


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i", "--input", type=str, required=True, help="Input JSON file or directory."
)
@click.option("-o", "--out_dir", default="./output", type=str, help="Output directory.")
@click.option(
    "-s",
    "--seeds",
    type=str,
    default=None,
    help="Seeds (comma-separated). Overrides JSON modelSeeds. "
    "If unset, uses modelSeeds from the input JSON, or a random seed when absent.",
)
@click.option("-c", "--cycle", type=int, default=10, help="Pairformer cycle number.")
@click.option("-p", "--step", type=int, default=200, help="Diffusion steps.")
@click.option("-e", "--sample", type=int, default=5, help="Number of samples.")
@click.option(
    "-d",
    "--dtype",
    type=str,
    default="fp32",
    help="Inference dtype. Defaults to fp32; pass bf16 to opt in.",
)
@click.option(
    "-n",
    "--model_name",
    type=str,
    default=DEFAULT_MODEL_NAME,
    help="Model checkpoint name.",
)
@click.option(
    "--load_checkpoint_path",
    type=str,
    default="",
    help="Explicit model checkpoint path. If unset, uses the released checkpoint filename for model_name.",
)
@click.option(
    "--use_msa",
    type=bool,
    default=True,
    help="Whether to use MSA for inference.",
)
@click.option(
    "--use_default_params",
    type=bool,
    default=False,
    help="Reset --cycle/--step to model defaults; currently redundant for opendde_v1.",
)
@click.option(
    "--trimul_kernel",
    type=str,
    default="auto",
    help="Triangle multiplicative update kernel ('auto', 'cuequivariance', or 'torch').",
)
@click.option(
    "--triatt_kernel",
    type=str,
    default="auto",
    help="Triangle attention kernel ('auto', 'cuequivariance', or 'torch').",
)
@click.option(
    "--enable_cache",
    type=bool,
    default=True,
    help="Cache shareable variables in the diffusion module.",
)
@click.option(
    "--enable_fusion",
    type=bool,
    default=True,
    help="Enable efficient kernel fusion in the diffusion transformer.",
)
@click.option(
    "--enable_tf32",
    type=bool,
    default=True,
    help="Enable TF32 for FP32 matrix multiplications.",
)
@click.option(
    "--deterministic",
    type=bool,
    default=False,
    help="Enable deterministic PyTorch algorithms for reproducible inference.",
)
@click.option(
    "--use_template",
    type=bool,
    default=False,
    help="Use templates (requires templatesPath in input JSON).",
)
@click.option(
    "--use_rna_msa",
    type=bool,
    default=False,
    help="Use RNA MSA (requires rnaSequence.unpairedMsaPath in input JSON).",
)
@click.option(
    "--msa_server_mode",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated compatibility option; ignored.",
)
@click.option(
    "--need_atom_confidence",
    type=bool,
    default=False,
    help="Whether to compute atom-level confidence scores.",
)
@click.option(
    "--foldcp_mode",
    type=click.Choice(["single", "distributed"]),
    default="single",
    help="Fold-CP execution mode: original single-card path or distributed CP path.",
)
@click.option(
    "--foldcp_size_dp",
    type=int,
    default=1,
    help="Number of data-parallel ranks for Fold-CP.",
)
@click.option(
    "--foldcp_size_cp",
    type=int,
    default=1,
    help="Number of context-parallel ranks; distributed mode requires a square value such as 4.",
)
@click.option(
    "--foldcp_devices",
    type=str,
    default="",
    help="Optional visible device list recorded in Fold-CP metrics.",
)
@click.option(
    "--foldcp_metrics_jsonl",
    type=str,
    default="",
    help="Optional JSONL path for Fold-CP speed/memory records.",
)
@click.option(
    "--kalign_binary_path",
    type=str,
    default=None,
    help="Path to kalign (searches in PATH if not provided).",
)
@click.option(
    "--prot_template_mmcif_dir",
    type=str,
    default=None,
    help="Directory for protein template mmCIF files. Overrides the default "
    "$OPENDDE_ROOT_DIR/search_database/mmcif; use a writable path when the data "
    "root is read-only and templates are fetched on demand.",
)
@click.option(
    "--prot_template_cache_dir",
    type=str,
    default=None,
    help="Directory for cached parsed protein templates. Overrides the default "
    "$OPENDDE_ROOT_DIR/search_database/template_cache; use a writable path when "
    "the data root is read-only.",
)
@click.option(
    "--use_tfg_guidance",
    type=bool,
    default=False,
    help="Use Training-Free Guidance (TFG) for inference.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch (searches in PATH if not provided).",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild (searches in PATH if not provided).",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
def predict(
    input: str,
    out_dir: str,
    seeds: Optional[str],
    cycle: int,
    step: int,
    sample: int,
    dtype: Literal["bf16", "fp32"],
    model_name: str,
    load_checkpoint_path: str,
    use_msa: bool,
    use_default_params: bool,
    trimul_kernel: str,
    triatt_kernel: str,
    enable_cache: bool,
    enable_fusion: bool,
    enable_tf32: bool,
    deterministic: bool,
    use_template: bool,
    use_rna_msa: bool,
    msa_server_mode: Optional[str],
    need_atom_confidence: bool,
    kalign_binary_path: Optional[str] = None,
    prot_template_mmcif_dir: Optional[str] = None,
    prot_template_cache_dir: Optional[str] = None,
    use_tfg_guidance: bool = False,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
    nhmmer_binary_path: Optional[str] = None,
    hmmalign_binary_path: Optional[str] = None,
    hmmbuild_rna_binary_path: Optional[str] = None,
    ntrna_database_path: Optional[str] = None,
    rfam_database_path: Optional[str] = None,
    rna_central_database_path: Optional[str] = None,
    nhmmer_n_cpu: Optional[int] = None,
    foldcp_mode: Literal["single", "distributed"] = "single",
    foldcp_size_dp: int = 1,
    foldcp_size_cp: int = 1,
    foldcp_devices: str = "",
    foldcp_metrics_jsonl: str = "",
) -> None:
    """
    Run predictions with OpenDDE using various input formats.

    Args:
        input (str): Input JSON file or directory.
        out_dir (str): Output directory for results.
        seeds (Optional[str]): Comma-separated seeds; overrides JSON modelSeeds.
            When None, falls back to JSON modelSeeds or a random seed.
        cycle (int): Number of cycles.
        step (int): Number of diffusion steps.
        sample (int): Number of samples.
        dtype (str): Data type.
        model_name (str): Model name.
        load_checkpoint_path (str): Explicit checkpoint path.
        use_msa (bool): Use MSA.
        use_default_params (bool): Reset cycle/step to model defaults.
        trimul_kernel (str): Kernel for triangle multiplicative.
        triatt_kernel (str): Kernel for triangle attention.
        enable_cache (bool): Enable shared variables cache.
        enable_fusion (bool): Enable efficient fusion.
        enable_tf32 (bool): Enable TF32.
        deterministic (bool): Enable deterministic PyTorch algorithms.
        use_template (bool): Use templates.
        use_rna_msa (bool): Use RNA MSA.
        msa_server_mode (Optional[str]): Deprecated compatibility option; ignored.
        need_atom_confidence (bool): Compute atom-level confidence scores.
        kalign_binary_path (Optional[str]): Path to kalign binary.
        prot_template_mmcif_dir (Optional[str]): Override for the protein template
            mmCIF directory; use a writable path when the data root is read-only.
        prot_template_cache_dir (Optional[str]): Override for the parsed template
            cache directory; use a writable path when the data root is read-only.
        use_tfg_guidance (bool): Use TFG guidance.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): NT-RNA database path.
        rfam_database_path (Optional[str]): Rfam database path.
        rna_central_database_path (Optional[str]): RNAcentral database path.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        foldcp_mode (str): Fold-CP execution mode.
        foldcp_size_dp (int): Number of data-parallel ranks.
        foldcp_size_cp (int): Number of context-parallel ranks.
        foldcp_devices (str): Optional visible device list recorded in metrics.
        foldcp_metrics_jsonl (str): Optional JSONL path for benchmark records.
    """
    init_logging()
    logger.info(f"Run infer with input={input}, out_dir={out_dir}, sample={sample}")
    if use_default_params:
        if model_name in SUPPORTED_MODELS:
            cycle = 10
            step = 200
        else:
            raise RuntimeError(
                f"{model_name} is not supported for inference in our model list"
            )
    logger.info(
        f"Using inference params for model {model_name}: "
        f"cycle={cycle}, step={step}, use_msa={use_msa}"
    )
    assert trimul_kernel in [
        "auto",
        "cuequivariance",
        "torch",
    ], "Invalid trimul_kernel. Options: 'auto', 'cuequivariance', 'torch'."
    assert triatt_kernel in [
        "auto",
        "cuequivariance",
        "torch",
    ], "Invalid triatt_kernel. Options: 'auto', 'cuequivariance', 'torch'."
    # None => not provided on the command line; let inference fall back to JSON
    # modelSeeds (or a random seed) instead.
    seed_list = [int(s) for s in seeds.split(",")] if seeds else None

    if use_template:
        assert model_name in SUPPORTED_MODELS, (
            f"Only {', '.join(SUPPORTED_MODELS)} support template inference."
        )
        logger.info("=" * 50)
        logger.info(
            "Using templates for inference. Template files should have "
            ".hhr or .a3m extensions and be specified in the JSON file.\n"
            "Example: /path/to/template.hhr or /path/to/template.a3m\n"
            "Note: Inference will proceed with automatic template search "
            "if none are provided and use_template is True."
        )
        logger.info("=" * 50)

    if use_rna_msa:
        assert model_name in SUPPORTED_MODELS, (
            f"Only {', '.join(SUPPORTED_MODELS)} support RNA MSA inference."
        )
        logger.info("=" * 50)
        logger.info(
            "Using RNA MSA for inference. RNA MSA files should have .a3m "
            "extension and be specified in the JSON file.\n"
            "Example: /path/to/rna_msa.a3m\n"
            "Note: Inference will proceed with automatic RNA MSA search "
            "if none are provided and use_rna_msa is True."
        )
        logger.info("=" * 50)

    if use_tfg_guidance:
        logger.info("Using Training-Free Guidance (TFG) for inference.")

    inference_jsons(
        input,
        out_dir,
        use_msa,
        seeds=seed_list,
        n_cycle=cycle,
        n_step=step,
        n_sample=sample,
        dtype=dtype,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        trimul_kernel=trimul_kernel,
        triatt_kernel=triatt_kernel,
        enable_cache=enable_cache,
        enable_fusion=enable_fusion,
        enable_tf32=enable_tf32,
        deterministic=deterministic,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        msa_server_mode=msa_server_mode,
        need_atom_confidence=need_atom_confidence,
        kalign_binary_path=kalign_binary_path,
        prot_template_mmcif_dir=prot_template_mmcif_dir,
        prot_template_cache_dir=prot_template_cache_dir,
        use_tfg_guidance=use_tfg_guidance,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
        foldcp_mode=foldcp_mode,
        foldcp_size_dp=foldcp_size_dp,
        foldcp_size_cp=foldcp_size_cp,
        foldcp_devices=foldcp_devices,
        foldcp_metrics_jsonl=foldcp_metrics_jsonl,
    )


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="PDB/CIF files or directory to generate inference JSONs.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option(
    "--altloc",
    default="first",
    type=str,
    help=(
        "Select the first altloc conformation of each residue, "
        "or specify the altloc letter ('A', 'B', etc.)."
    ),
)
@click.option(
    "--assembly_id",
    default=None,
    type=str,
    help="Assembly ID for structure extension (default: no extension).",
)
@click.option(
    "--include_discont_poly_poly_bonds",
    default=False,
    is_flag=True,
    help="Whether to include discontinuous polymer-polymer bonds.",
)
def tojson(
    input: str,
    out_dir: str = "./output",
    altloc: str = "first",
    assembly_id: Optional[str] = None,
    include_discont_poly_poly_bonds: bool = False,
) -> List[str]:
    """
    Convert PDB or CIF files to JSON files for OpenDDE inference.

    Args:
        input (str): Input PDB/CIF file or directory.
        out_dir (str): Output directory for JSON files.
        altloc (str): Alternate location conformation selection.
        assembly_id (Optional[str]): Assembly ID for structure extension.
        include_discont_poly_poly_bonds (bool): Whether to include discontinuous polymer-polymer bonds.

    Returns:
        List[str]: List of generated JSON file paths.
    """
    init_logging()
    logger.info(
        f"Run tojson with input={input}, out_dir={out_dir}, "
        f"altloc={altloc}, assembly_id={assembly_id}"
        f", include_discont_poly_poly_bonds={include_discont_poly_poly_bonds}"
    )
    input_files = []
    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} not exists.")
    if os.path.isdir(input):
        input_files.extend(
            [str(file) for file in Path(input).rglob("*") if file.is_file()]
        )
    elif os.path.isfile(input):
        input_files.append(input)
    else:
        raise RuntimeError(f"can not read a special file: {input}")

    input_files = [
        file for file in input_files if file.endswith(".pdb") or file.endswith(".cif")
    ]
    if len(input_files) == 0:
        raise RuntimeError(f"can not read a valid `pdb` or `cif` file from {input}")
    logger.info(
        f"will tojson jsons for {len(input_files)} input files with `pdb` or `cif` format."
    )
    output_jsons = []
    os.makedirs(out_dir, exist_ok=True)
    for input_file in input_files:
        stem, _ = os.path.splitext(os.path.basename(input_file))
        pdb_name = stem[:20]
        output_json = os.path.join(out_dir, f"{pdb_name}.json")
        if input_file.endswith(".pdb"):
            with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
                tmp_cif_file = tmp.name
                pdb_to_cif(input_file, tmp_cif_file)
                cif_to_input_json(
                    tmp_cif_file,
                    assembly_id=assembly_id,
                    altloc=altloc,
                    sample_name=pdb_name,
                    output_json=output_json,
                    include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
                )
        elif input_file.endswith(".cif"):
            cif_to_input_json(
                input_file,
                assembly_id=assembly_id,
                altloc=altloc,
                output_json=output_json,
                include_discont_poly_poly_bonds=include_discont_poly_poly_bonds,
            )
        else:
            raise RuntimeError(f"can not read a special ligand_file: {input_file}")
        output_jsons.append(output_json)
    logger.info(f"{len(output_jsons)} generated jsons have been saved to {out_dir}.")
    return output_jsons


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON or FASTA file for MSA search.",
)
@click.option("-o", "--out_dir", type=str, default="./output", help="Output directory.")
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated compatibility option; ignored.",
)
def msa(input: str, out_dir: str, msa_server_mode: Optional[str]) -> Union[str, dict]:
    """
    Perform MSA search using MMseqs2.
    If input is a FASTA file, it should contain protein sequences.

    Args:
        input (str): Path to a JSON or FASTA file.
        out_dir (str): Directory to save MSA results.
        msa_server_mode (Optional[str]): Deprecated compatibility option; ignored.

    Returns:
        Union[str, dict]: Updated JSON path or dictionary of MSA results.
    """
    init_logging()
    logger.info(f"Run msa with input={input}, out_dir={out_dir}")
    if input.endswith(".json"):
        msa_input_json, _ = update_infer_json(
            input, out_dir, use_msa=True, mode=msa_server_mode
        )
        logger.info(f"msa results have been update to {msa_input_json}")
        return msa_input_json
    elif input.endswith(".fasta"):
        records = list(SeqIO.parse(input, "fasta"))
        protein_seqs = []
        for seq in records:
            protein_seqs.append(str(seq.seq))
        protein_seqs = sorted(protein_seqs)
        msa_res_subdirs = msa_search(protein_seqs, out_dir, mode=msa_server_mode)
        assert len(msa_res_subdirs) == len(protein_seqs), "msa search failed"
        fasta_msa_res = dict(zip(protein_seqs, msa_res_subdirs))
        logger.info(
            f"msa result is: {fasta_msa_res}, and it has been save to {out_dir}"
        )
        return fasta_msa_res
    else:
        raise RuntimeError(f"only support `json` or `fasta` format, but got : {input}")


# The new msatemplate command first performs MSA search, then performs template search
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON file for MSA and template search.",
)
@click.option(
    "-o",
    "--out_dir",
    type=str,
    default="./output",
    help="Output directory.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch (searches in PATH if not provided).",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild (searches in PATH if not provided).",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated compatibility option; ignored.",
)
def msatemplate(
    input: str,
    out_dir: str,
    hmmsearch_binary_path: Optional[str],
    hmmbuild_binary_path: Optional[str],
    seqres_database_path: Optional[str],
    msa_server_mode: Optional[str],
) -> str:
    """
    Perform MSA search followed by template search.

    Args:
        input (str): Path to the input JSON file.
        out_dir (str): Directory to save MSA and template results.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        msa_server_mode (Optional[str]): Deprecated compatibility option; ignored.

    Returns:
        str: Updated JSON file path with template information.
    """
    logger.info(f"Run msa_template with input={input}, out_dir={out_dir}")

    if not input.endswith(".json"):
        raise RuntimeError(
            f"msa_template only supports `json` format, but got: {input}"
        )

    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} does not exist")

    return preprocess_input(
        input_json=input,
        out_dir=out_dir,
        use_msa=True,
        use_template=True,
        use_rna_msa=False,
        msa_server_mode=msa_server_mode,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
    )


# The new inputprep command calls the RNA MSA process after the MSA template process finishes
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=str,
    required=True,
    help="JSON file to update with RNA MSA (supports 'json' format only).",
)
@click.option(
    "-o",
    "--out_dir",
    type=str,
    default="./output",
    help="Output directory.",
)
@click.option(
    "--hmmsearch_binary_path",
    type=str,
    default=None,
    help="Path to hmmsearch.",
)
@click.option(
    "--hmmbuild_binary_path",
    type=str,
    default=None,
    help="Path to hmmbuild.",
)
@click.option(
    "--seqres_database_path",
    type=str,
    default=None,
    help="Path to the sequence database for template search.",
)
@click.option(
    "--nhmmer_binary_path",
    type=str,
    default=None,
    help="Path to nhmmer for RNA MSA search.",
)
@click.option(
    "--hmmalign_binary_path",
    type=str,
    default=None,
    help="Path to hmmalign for RNA MSA search.",
)
@click.option(
    "--hmmbuild_rna_binary_path",
    type=str,
    default=None,
    help="Path to RNA-specific hmmbuild.",
)
@click.option(
    "--ntrna_database_path",
    type=str,
    default=None,
    help="Path to the NT-RNA database.",
)
@click.option(
    "--rfam_database_path",
    type=str,
    default=None,
    help="Path to the Rfam database.",
)
@click.option(
    "--rna_central_database_path",
    type=str,
    default=None,
    help="Path to the RNAcentral database.",
)
@click.option(
    "--nhmmer_n_cpu",
    type=int,
    default=None,
    help="Number of CPUs for nhmmer.",
)
@click.option(
    "-m",
    "--msa_server_mode",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated compatibility option; ignored.",
)
def inputprep(
    input: str,
    out_dir: str,
    hmmsearch_binary_path: Optional[str],
    hmmbuild_binary_path: Optional[str],
    seqres_database_path: Optional[str],
    nhmmer_binary_path: Optional[str],
    hmmalign_binary_path: Optional[str],
    hmmbuild_rna_binary_path: Optional[str],
    ntrna_database_path: Optional[str],
    rfam_database_path: Optional[str],
    rna_central_database_path: Optional[str],
    nhmmer_n_cpu: Optional[int],
    msa_server_mode: Optional[str],
) -> str:
    """
    Perform MSA search, template search, and RNA MSA search sequentially.

    Args:
        input (str): Path to the input JSON file.
        out_dir (str): Directory to save all search results.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.
        nhmmer_binary_path (Optional[str]): Path to nhmmer binary.
        hmmalign_binary_path (Optional[str]): Path to hmmalign binary.
        hmmbuild_rna_binary_path (Optional[str]): Path to RNA hmmbuild binary.
        ntrna_database_path (Optional[str]): Path to NT-RNA database.
        rfam_database_path (Optional[str]): Path to Rfam database.
        rna_central_database_path (Optional[str]): Path to RNAcentral database.
        nhmmer_n_cpu (Optional[int]): Number of CPUs for nhmmer.
        msa_server_mode (Optional[str]): Deprecated compatibility option; ignored.

    Returns:
        str: Final updated JSON file path with all search information.
    """
    logger.info(f"Run inputprep with input={input}, out_dir={out_dir}")

    if not input.endswith(".json"):
        raise RuntimeError(f"inputprep only supports `json` format, but got: {input}")

    if not os.path.exists(input):
        raise RuntimeError(f"input file {input} does not exist")

    return preprocess_input(
        input_json=input,
        out_dir=out_dir,
        use_msa=True,
        use_template=True,
        use_rna_msa=True,
        msa_server_mode=msa_server_mode,
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        seqres_database_path=seqres_database_path,
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmbuild_rna_binary_path=hmmbuild_rna_binary_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        nhmmer_n_cpu=nhmmer_n_cpu,
    )


opendde_cli.add_command(predict, name="pred")
opendde_cli.add_command(doctor, name="doctor")
opendde_cli.add_command(tojson, name="json")
opendde_cli.add_command(msa, name="msa")
opendde_cli.add_command(msatemplate, name="mt")
opendde_cli.add_command(inputprep, name="prep")

if __name__ == "__main__":
    opendde_cli()
