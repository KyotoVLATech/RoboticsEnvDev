# 環境構築手順
- このレポジトリがサブモジュールを含めてクローンされていることを確認してください `git clone --recursive ~`
- 仮想環境の作成 `uv sync`
- 仮想環境の有効化 `source .venv/bin/activate`
- `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- genesisのインストール `uv pip install -e ./Genesis/`
- lerobotのインストール `uv pip install -e ./lerobot/`