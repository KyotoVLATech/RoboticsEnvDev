# 環境構築手順
```bash
git clone --recursive https://github.com/KyotoVLATech/RoboticsEnvDev.git && cd RoboticsEnvDev
uv sync
```
# シミュレーション
## データセット作成
```bash
uv run scripts/make_sim_dataset.py
```