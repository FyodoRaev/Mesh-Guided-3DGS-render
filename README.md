# Distillate

Минимальный mesh + 3DGS hybrid.

Mesh во время обучения берётся из `scene_dir/mesh_support/*.npz`, если кеш полный. Если кеша нет или он неполный, `train.py` рендерит mesh live через тот же `MeshRenderer`.

## Precompute mesh

```bash
python precompute_mesh_support.py \
  --scene_dir ../scene_yellow_car \
  --mesh_obj ../scene_yellow_car/yellow_car.obj
```

Скрипт сохраняет `rgb/depth/mask`, сразу перечитывает `.npz` через `MeshSupportCache` и падает, если cache != live slow render. Первые кадры получают `check_*.png`: live | cache | diff.

## Hybrid train

```bash
python train.py \
  --scene_dir ../scene_yellow_car \
  --mesh_obj ../scene_yellow_car/yellow_car.obj \
  --result_dir runs/hybrid_simple
```

Главные флаги: `--mesh_support_dir`, `--force_live_mesh`, `--init_means_ckpt`, `--init_points`, `--max_steps`, `--eval_every`, `--save_every`, `--max_gs`, `--gate_eps`, `--gate_band`.

Pure 3DGS baseline оставлен отдельно: `train_gs_only.py`.
