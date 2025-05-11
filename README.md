# 環境構築手順
- このレポジトリがサブモジュールを含めてクローンされていることを確認してください `git clone --recursive git@github.com:KyotoVLATech/RoboticsEnvDev.git`
- サブモジュールのアップデートを行ってください `git submodule update --remote`
- 仮想環境の作成 `uv sync`
- 仮想環境の有効化 `source .venv/bin/activate`
- genesisのインストール `uv pip install -e ./Genesis/`
- lerobotのインストール `uv pip install -e ./lerobot/`

## TODO
- [ ] MRによるマスタースレーブ検証用に、aloha用のGenesis環境を作る
- [ ] aloha環境に布を配置する
- [ ] cubeBが最初から箱に入っている問題に対処する
- [ ] aloha制御用のMRソフトを開発する
- [ ] eval_policy.pyとmake_sim_dataset.pyをインライン引数に対応させる