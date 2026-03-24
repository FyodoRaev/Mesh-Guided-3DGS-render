# Distillate (minimal)

Файлы:
- `train.py` — основной trainer
- `colmap_data.py` — загрузка COLMAP txt + изображений
- `mesh_renderer.py` — mesh RGB/depth/mask
- `hybrid_math.py` — формулы compose/gate/weight
- `diagnostics.py` — сохранение панелей и heatmap
- `selfcheck.py` — быстрый sanity-check

## Запуск

```bash
cd /home/agisoft/PycharmProjects/HybridGSMesh/distillate
source .venv-gs3090/bin/activate
python train.py \
  --scene_dir ../scene \
  --mesh_obj ../scene/yellow_car.obj \
  --result_dir ../results/distillate_run_001 \
  --max_steps 6000 \
  --eval_every 2000 \
  --save_every 2000 \
  --batch_size 1 \
  --num_workers 0
```

## Визуализация (минимальными усилиями)

- Валид. кадры и карты: `results/.../vis/step_xxxxxx/`
- Train-снапшоты: `results/.../train_vis/step_xxxxxx/`

Полезные флаги:
- `--save_vis_images 4` — сколько val-кадров сохранять на каждом eval
- `--save_train_vis_every 1000` — как часто сохранять train snapshot (0 = выключено)

Сохраняются:
- `gt`, `mesh`, `gs`, `hybrid`
- `weight`, `gate`, `mesh_mask`
- `mesh_err`, `gs_err`, `hybrid_err`
- `better_than_mesh`, `worse_than_mesh`
- `panel` (всё вместе)

## Проверка

```bash
cd /home/agisoft/PycharmProjects/HybridGSMesh/distillate
source .venv-gs3090/bin/activate
python selfcheck.py
```
