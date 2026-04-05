# Distillate (radically simplified)

Минимальный пайплайн из двух шагов:

1. `positions_pretrain_minimal.py` — pretrain только `means` (притягивание проекций к bad-зонам mesh).
2. `train.py` — hybrid GS+mesh training с инициализацией `means` из шага 1.

## Файлы

- `positions_pretrain_minimal.py` — pretrain позиций + визуализации `before/after/bad`
- `train.py` — hybrid training + eval
- `diagnostics.py` — 4 eval-визуализации: `mesh`, `gs`, `hybrid`, `compare`
- `hybrid_math.py` — глубинный gate + hybrid compose + PSNR
- `colmap_data.py` — загрузка COLMAP
- `mesh_renderer.py` — mesh render (rgb/depth/mask)
- `selfcheck.py` — sanity-check математики

## Шаг 1: pretrain means

```bash
cd /home/agisoft/PycharmProjects/HybridGSMesh/distillate
source .venv-gs3090/bin/activate

python positions_pretrain_minimal.py \
  --scene_dir ../scene \
  --mesh_obj ../scene/yellow_car.obj \
  --result_dir runs/positions_pretrain_minimal
```

Артефакты:
- `runs/positions_pretrain_minimal/ckpts/ckpt_*.pt`
- `runs/positions_pretrain_minimal/stats/eval_*.json`
- `runs/positions_pretrain_minimal/vis/step_*/{train,val}_*_before_after.png`

## Шаг 2: hybrid training

```bash
python train.py \
  --scene_dir ../scene \
  --mesh_obj ../scene/yellow_car.obj \
  --init_means_ckpt runs/positions_pretrain_minimal/ckpts/ckpt_001000.pt \
  --result_dir runs/hybrid_from_pretrain
```

Артефакты:
- `runs/hybrid_from_pretrain/ckpts/ckpt_*.pt`
- `runs/hybrid_from_pretrain/stats/eval_*.json`
- `runs/hybrid_from_pretrain/vis/step_*/0000_mesh.png`
- `runs/hybrid_from_pretrain/vis/step_*/0000_gs.png`
- `runs/hybrid_from_pretrain/vis/step_*/0000_hybrid.png`
- `runs/hybrid_from_pretrain/vis/step_*/0000_compare.png`

## Проверка

```bash
python selfcheck.py
python -m py_compile *.py
```
