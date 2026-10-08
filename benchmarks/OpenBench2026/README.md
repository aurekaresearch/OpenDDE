# OpenBench2026

Evaluation panel built from PDB entries released between 2026-01-07 and
2026-08-12. Each target is the first biological assembly of an entry
(`<pdb_id>-assembly1`); the interfaces in `interfaces.csv` are the ones scored.

| Panel | Interfaces | Targets | Content |
| --- | ---: | ---: | --- |
| [`abag`](abag) | 316 | 206 | antibody–antigen interfaces |

Only assemblies of at most 1,600 tokens are kept.

The panel folder holds

- `targets.txt`: one target id per line;
- `interfaces.csv`: one row per scored interface. `interface_chain_id_1` and
  `interface_chain_id_2` are the two chains of the interface; `antibody_chain`
  and `antigen_chain` say which is which.

The 2026 antibody–protein common release is listed in
[`../2026ARK_AB`](../2026ARK_AB).
