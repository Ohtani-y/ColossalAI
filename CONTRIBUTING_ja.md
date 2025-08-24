# Colossal-AI コントリビューションガイド

Colossal-AIプロジェクトへの貢献にご興味をお持ちいただき、ありがとうございます！このガイドでは、プロジェクトに貢献するための手順とガイドラインを説明します。

## 目次

- [開発環境のセットアップ](#開発環境のセットアップ)
- [コーディング規約](#コーディング規約)
- [貢献の種類](#貢献の種類)
- [貢献の流れ](#貢献の流れ)
- [プルリクエストのガイドライン](#プルリクエストのガイドライン)
- [テストの実行](#テストの実行)
- [ドキュメントの貢献](#ドキュメントの貢献)
- [コミュニティガイドライン](#コミュニティガイドライン)

## 開発環境のセットアップ

### 前提条件
- Python 3.8以上
- CUDA 11.0以上（GPU使用時）
- Git

### セットアップ手順

1. **既存のColossal-AIをアンインストール**
   ```bash
   pip uninstall colossalai
   ```

2. **リポジトリをフォークしてクローン**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ColossalAI.git
   cd ColossalAI
   ```

3. **upstreamリモートを追加**
   ```bash
   git remote add upstream https://github.com/hpcaitech/ColossalAI.git
   ```

4. **開発モードでインストール**
   ```bash
   # CUDA拡張なし（推奨：開発時）
   pip install -e .
   
   # CUDA拡張あり（オプション）
   CUDA_EXT=1 pip install -e .
   ```

5. **開発依存関係をインストール**
   ```bash
   pip install -r requirements/requirements-dev.txt
   pip install -r requirements/requirements-test.txt
   ```

## コーディング規約

### コードスタイル
- **Python**: PEP 8に準拠
- **行の長さ**: 最大120文字
- **インデント**: スペース4つ
- **命名規則**: 
  - 変数・関数: snake_case
  - クラス: PascalCase
  - 定数: UPPER_CASE

### pre-commitフックの設定
```bash
pip install pre-commit
pre-commit install
```

これにより、コミット時に自動的にコードスタイルがチェックされます。

### 型ヒント
新しいコードには型ヒントを追加してください：
```python
def process_data(data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, int]:
    # 実装
    pass
```

## 貢献の種類

### バグ修正
- 既存のバグを修正する
- テストケースを追加する
- 修正内容を明確に説明する

### 新機能の追加
- 新しい並列化戦略
- 新しいモデルサポート
- パフォーマンス最適化

### ドキュメントの改善
- APIドキュメントの更新
- チュートリアルの追加
- 翻訳作業

### テストの改善
- テストカバレッジの向上
- 新しいテストケースの追加
- CI/CDの改善

## 貢献の流れ

### 1. Issueの確認・作成
- 既存のIssueを確認し、重複を避ける
- 新しい機能やバグ修正の場合は、まずIssueを作成して議論する

### 2. ブランチの作成
```bash
git fetch upstream
git checkout main
git merge upstream/main
git checkout -b feature/your-feature-name
```

### 3. 開発作業
- 小さな、論理的なコミットに分割する
- 明確なコミットメッセージを書く
- テストを追加・更新する

### 4. テストの実行
```bash
# 全テストの実行
pytest tests/

# CPUのみのテスト
pytest -m cpu tests/

# 特定のテストファイル
pytest tests/test_specific_module.py
```

### 5. プルリクエストの作成
- 変更内容を明確に説明する
- 関連するIssueをリンクする
- レビューアーを指定する

## プルリクエストのガイドライン

### PRタイトル
- 簡潔で説明的なタイトルを使用
- 例：`[Feature] Add DeepSeek R1 GRPO support`、`[Fix] Resolve memory leak in tensor parallel`

### PR説明
以下の項目を含めてください：
- **概要**: 変更内容の簡潔な説明
- **動機**: なぜこの変更が必要か
- **変更内容**: 具体的な変更点
- **テスト**: 追加・変更されたテスト
- **チェックリスト**: 完了した項目にチェック

### PRテンプレート例
```markdown
## 概要
DeepSeek R1モデルのGRPO学習サポートを追加

## 動機
数学的推論能力向上のため、DeepSeek R1のGRPO学習機能が必要

## 変更内容
- GRPOトレーナーの実装
- DeepSeek R1モデル設定の追加
- 学習スクリプトとドキュメントの更新

## テスト
- [ ] 単体テストの追加
- [ ] 統合テストの実行
- [ ] パフォーマンステストの実行

## チェックリスト
- [ ] コードスタイルチェック通過
- [ ] テスト追加・更新
- [ ] ドキュメント更新
- [ ] 後方互換性の確認
```

## テストの実行

### ローカルテスト
```bash
# 全テストの実行
pytest tests/

# 並列テスト（高速化）
pytest -n auto tests/

# カバレッジレポート
pytest --cov=colossalai tests/

# 特定のマーカーでテスト
pytest -m "not slow" tests/
```

### GPU テスト
```bash
# GPU必須のテスト
pytest -m gpu tests/

# 分散テスト
torchrun --nproc_per_node=2 -m pytest tests/test_distributed/
```

### CI/CD
プルリクエスト作成時に自動的に以下が実行されます：
- コードスタイルチェック
- 単体テスト
- 統合テスト
- ドキュメントビルド

## ドキュメントの貢献

### ドキュメントの種類
- **APIドキュメント**: docstringの更新
- **チュートリアル**: 新機能の使用方法
- **ガイド**: ベストプラクティス
- **翻訳**: 多言語サポート

### ドキュメントのビルド
```bash
cd docs
pip install -r requirements.txt
make html
```

### 翻訳作業
- 英語ドキュメントを`docs/source/en/`に配置
- 日本語翻訳を`docs/source/ja/`に配置
- 中国語翻訳を`docs/source/zh-Hans/`に配置

## コミュニティガイドライン

### 行動規範
- 敬意を持って接する
- 建設的なフィードバックを提供する
- 多様性を尊重する
- 学習と成長を支援する

### コミュニケーション
- **GitHub Issues**: バグ報告、機能リクエスト
- **GitHub Discussions**: 一般的な質問、アイデア共有
- **Slack**: リアルタイムコミュニケーション
- **WeChat**: 中国語コミュニティ

### レビュープロセス
1. **自動チェック**: CI/CDによる自動テスト
2. **コードレビュー**: メンテナーによるレビュー
3. **テスト**: 機能テストとパフォーマンステスト
4. **承認**: 最終承認とマージ

## よくある質問

### Q: 初回貢献者です。何から始めればよいですか？
A: `good first issue`ラベルの付いたIssueから始めることをお勧めします。

### Q: 大きな機能を追加したいのですが？
A: まずIssueを作成して設計について議論してください。RFC（Request for Comments）プロセスを経る場合があります。

### Q: テストが失敗します。どうすればよいですか？
A: エラーメッセージを確認し、ローカルで再現してください。解決できない場合は、コミュニティに質問してください。

### Q: ドキュメントのみの変更でもPRを作成できますか？
A: はい！ドキュメントの改善も重要な貢献です。

## サポート

貢献に関してご質問がある場合は、以下の方法でお気軽にお問い合わせください：

- [GitHub Discussions](https://github.com/hpcaitech/ColossalAI/discussions)
- [Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
- Email: contact@hpc-ai.tech

皆様の貢献をお待ちしております！
