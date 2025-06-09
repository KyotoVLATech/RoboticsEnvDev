# 環境構築手順
- このレポジトリがサブモジュールを含めてクローンされていることを確認してください `git clone --recursive git@github.com:KyotoVLATech/RoboticsEnvDev.git`
- サブモジュールのアップデートを行ってください `git submodule update --remote`
- 仮想環境の作成 `uv sync`
- 仮想環境の有効化 `source .venv/bin/activate`
- genesisのインストール `uv pip install -e "Genesis"`
- lerobotのインストール `uv pip install -e "lerobot/[feetech, aloha, smolvla]`

## キャリブレーション
[こちらの動画](https://huggingface.co/docs/lerobot/en/so101#calibration-video)を参照
- follower
```bash
uv run -m lerobot.calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so100_black_follower_arm
```
- leader
```bash
uv run -m lerobot.calibrate \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=so100_black_leader_arm
```
## カメラの検出
```bash
python lerobot/lerobot/find_cameras.py opencv
```
使いたいカメラのpathを把握する
## テレオペレーション
```bash
uv run -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so100_black_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=so100_black_leader_arm \
    --display_data=true
```
## データセットのレコード
```bash
export DATASET_NAME=[データセット名]
uv run -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so100_black_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=so100_black_leader_arm \
    --display_data=true \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=dataset/${DATASET_NAME} \
    --dataset.num_episodes=100 \
    --dataset.single_task="Grab a handkerchief and open it"
```
- `--dataset.episode_time_s=60`
  Duration of each data recording episode (default: **60 seconds**).
- `--dataset.reset_time_s=60`
  Duration for resetting the environment after each episode (default: **60 seconds**).
- `--dataset.num_episodes=50`
  Total number of episodes to record (default: **50**).
- Press **Right Arrow (`→`)**: Early stop the current episode or reset time and move to the next.
- Press **Left Arrow (`←`)**: Cancel the current episode and re-record it.
## TODO
- [ ] aloha制御用のMRソフトを開発する
- [ ] eval_policy.pyとmake_sim_dataset.pyをインライン引数に対応させる