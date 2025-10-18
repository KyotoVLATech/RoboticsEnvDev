# RoboticsEnv (SO-101)
## システム要件
- Ubuntu 24.04
- CUDA対応のNvidia GPU
## 環境構築手順
```bash
git clone --recurse -b dev/so-101 https://github.com/KyotoVLATech/RoboticsEnvDev.git && cd RoboticsEnvDev
uv sync
uv pip install -e "Genesis"
uv pip install -e "lerobot/[smolvla, pi]"
uv pip uninstall torch torchvision
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
## データセット収集
```bash
uv run src/make_sim_dataset.py
```
収集するエピソード数などを変更できる
```python
main(episode_num=1, task=task, stage_dict=stage_dict, observation_height=480, observation_width=640, show_viewer=False)
```
## 学習
lerobotのReadmeやdocsを参照
## 評価
基本的にはsrc/eval_policy.pyを使うが，コードが古いので修正の必要がある