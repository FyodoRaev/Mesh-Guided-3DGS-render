# Distillate (minimal)

Файлы:
- `train.py` — основной trainer
- `colmap_data.py` — загрузка COLMAP txt + изображений
- `mesh_renderer.py` — mesh RGB/depth/mask
- `hybrid_math.py` — формулы compose/gate/weight
- `diagnostics.py` — сохранение минимальных диагностических карт
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

## Визуализация

- Валид. кадры и карты: `results/.../vis/step_xxxxxx/`

Полезный флаг:
- `--save_vis_images 4` — сколько val-кадров сохранять на каждом eval

Сохраняются:
- `gt`, `mesh`, `hybrid`
- `weight`, `gate`
- `better_than_mesh`, `worse_than_mesh`

## Проверка

```bash
cd /home/agisoft/PycharmProjects/HybridGSMesh/distillate
source .venv-gs3090/bin/activate
python selfcheck.py
```
