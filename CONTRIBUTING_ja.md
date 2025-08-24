# コントリビューションガイド

## 開発環境のセットアップ
1. 既存の Colossal-AI をアンインストールします。
   ```bash
   pip uninstall colossalai
   ```
2. リポジトリを取得します。
   ```bash
   git clone https://github.com/hpcaitech/ColossalAI.git
   cd ColossalAI
   ```
3. 開発モードでインストールします。
   ```bash
   pip install -e .
   ```

## コーディング規約
- テストに必要な依存関係をインストールします。
  ```bash
  pip install -r requirements/requirements-test.txt
  ```
- CPU のみのテストは次で実行できます。
  ```bash
  pytest -m cpu tests/
  ```
- コードスタイルチェックには pre-commit を使用します。
  ```bash
  pip install pre-commit
  pre-commit install
  ```

## 貢献の流れ
1. GitHub でリポジトリをフォークします。
2. upstream を設定し、最新の `main` を取得します。
   ```bash
   git remote add upstream https://github.com/hpcaitech/ColossalAI.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```
3. 作業用ブランチを作成します。
   ```bash
   git checkout -b feature/my-change
   ```
4. 変更をコミットしてプッシュします。
   ```bash
   git add -A
   git commit -m "Add my feature"
   git push origin feature/my-change
   ```
5. GitHub 上で Pull Request を作成します。関連する Issue があればリンクしてください。
