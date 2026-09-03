# Supported Models


OpenDDE currently exposes one public model:

| Model name | MSA / Constraint / RNA MSA / Template | Model Parameters (M) | Data cutoff |
| --- | :---: | ---: | :---: |
| `opendde_v1` | ✓ / × / ✓ / ✓ | 656 | 2021-09-30 |

Exact parameter count: `655,791,538`, rounded to `656 M`.
`opendde_v1` uses the defaults in `opendde/config/model_base.py`.

Use it with:

```bash
opendde pred -i examples/input.json -o ./output -n opendde_v1
```

Checkpoint path by default:

```text
$OPENDDE_ROOT_DIR/checkpoint/opendde.pt
```

## Released Checkpoints

| Checkpoint | Use case | Download |
| --- | --- | --- |
| `opendde.pt` | General-purpose OpenDDE checkpoint. | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde.pt) |
| `opendde_abag.pt` | ABAG-optimized checkpoint for antibody-antigen complexes. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt) |

The release source revision is
`eddd563ce96571f784012edd8f045181c8f8627d`. Exact checkpoint sizes and SHA-256
digests are recorded in the package-installed
[`opendde/config/model_manifest.json`](../opendde/config/model_manifest.json):

| Checkpoint | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `opendde.pt` | 2,625,249,069 | `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc` |
| `opendde_abag.pt` | 2,625,271,509 | `5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d` |

`opendde.pt` is the default checkpoint for `-n opendde_v1`. Keep the
ABAG-optimized checkpoint as `opendde_abag.pt` and pass it explicitly with
`--load_checkpoint_path`.

Recommended inference defaults:

- `model.N_cycle = 10`
- `sample_diffusion.N_step = 200`
- triangle kernels: `auto`

These are also the current `opendde pred` CLI defaults for `opendde_v1`.

Legacy `constraint` fields are ignored by the inference-only build. Use
`covalent_bonds` for supported covalent links.
