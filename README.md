# 環境構築手順
- このレポジトリがサブモジュールを含めてクローンされていることを確認してください `git clone --recursive git@github.com:KyotoVLATech/RoboticsEnvDev.git`
- サブモジュールのアップデートを行ってください `git submodule update --remote`
- 仮想環境の作成 `uv sync`
- 仮想環境の有効化 `source .venv/bin/activate`
- genesisのインストール `uv pip install -e "Genesis"`
- lerobotのインストール `uv pip install -e "lerobot/[feetech, aloha, smolvla]"`

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
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=so100_black_leader_arm \
    --display_data=true \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=datasets/${DATASET_NAME} \
    --dataset.num_episodes=100 \
    --dataset.single_task="Grab a handkerchief and open it"
```
`--dataset.episode_time_s=60` 各データ記録エピソードの長さ（デフォルト：60秒）。
`--dataset.reset_time_s=60` 各エピソードの後に環境をリセットするための時間（デフォルト：60秒）。
`--dataset.num_episodes=50` 記録するエピソードの総数（デフォルト：50）。
右矢印キー (→): 現在のエピソードまたはリセット時間を早期に終了し、次に進みます。
左矢印キー (←): 現在のエピソードをキャンセルし、再記録します。
## ポリシーの学習
ロボットを制御するためのポリシーを学習させるには、[`python lerobot/scripts/train.py`](https://www.google.com/search?q=../lerobot/scripts/train.py) スクリプトを使用します。いくつかの引数が必要です。以下にコマンドの例を示します。

```bash
export DATASET_NAME=[データセット名]
export POLICY=act
uv run -m lerobot.scripts.train \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=datasets/${DATASET_NAME} \
    --policy.type=${POLICY} \
    --output_dir=outputs/train/${POLICY}_${DATASET_NAME} \
    --job_name=${POLICY}_${DATASET_NAME} \
    --policy.device=cuda \
    --wandb.enable=true \
    --batch_size=8 \
    --steps=100000
```
1.  `--dataset.repo_id=${HF_USER}/so101_test` でデータセットを引数として指定しました。
2.  `policy.type=act` でポリシーを指定しました。これにより、[`configuration_act.py`](https://www.google.com/search?q=../lerobot/common/policies/act/configuration_act.py) から設定が読み込まれます。重要な点として、このポリシーは、データセットに保存されているロボットのモーターの状態数、モーターのアクション数、およびカメラ（例：`laptop` や `phone`）の数に自動的に適応します。
3.  Nvidia GPUで学習を行うため `policy.device=cuda` を指定しましたが、Appleシリコンで学習する場合は `policy.device=mps` を使用することもできます。
4.  学習プロットを可視化するために [Weights and Biases](https://docs.wandb.ai/quickstart) を使用するため、`wandb.enable=true` を指定しました。これはオプションですが、使用する場合は `wandb login` を実行してログインしていることを確認してください。

チェックポイントから学習を再開するコマンドの例を示します。
```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

- smolvla
```bash
export DATASET_NAME=[データセット名]
uv run -m lerobot.scripts.train \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=datasets/${DATASET_NAME} \
    --policy.path=lerobot/smolvla_base \
    --output_dir=outputs/train/smolvla_${DATASET_NAME} \
    --job_name=smolvla_${DATASET_NAME} \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --batch_size=64 \
    --steps=200000
```
学習後の`config.json`の`n_action_steps`が1になっているので編集して50にする
## ポリシーの評価

ポリシーのチェックポイントを入力として、[`lerobot/record.py`](https://www.google.com/search?q=%5Bhttps://github.com/huggingface/lerobot/blob/main/lerobot/record.py%5D\(https://github.com/huggingface/lerobot/blob/main/lerobot/record.py\)) の `record` スクリプトを使用できます。例えば、10個の評価エピソードを記録するには、次のコマンドを実行します。

```bash
export DATASET_NAME=eval_policy
uv run -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=so100_black_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video8, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=so100_black_leader_arm \
    --display_data=false \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=datasets/${DATASET_NAME} \
    --dataset.num_episodes=1 \
    --dataset.single_task="Grab a handkerchief and open it" \
    --policy.path=outputs/train/act_open-handkerchief/checkpoints/last/pretrained_model
```
# シミュレーション
## データセット作成
```bash
uv run src/make_sim_dataset.py
```

## TODO
- [ ] aloha制御用のMRソフトを開発する
- [ ] eval_policy.pyとmake_sim_dataset.pyをインライン引数に対応させる