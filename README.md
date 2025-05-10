# 環境構築手順
- このレポジトリがサブモジュールを含めてクローンされていることを確認してください `git clone --recursive git@github.com:KyotoVLATech/RoboticsEnvDev.git`
- サブモジュールのアップデートを行ってください `git submodule update --remote`
- 仮想環境の作成 `uv sync`
- 仮想環境の有効化 `source .venv/bin/activate`
- genesisのインストール `uv pip install -e ./Genesis/`
- lerobotのインストール `uv pip install -e ./lerobot/`
