# DeepSeek R1 統合ガイド

このガイドでは、Colossal-AIを使用してDeepSeek R1モデルを学習・推論する方法について詳しく説明します。

## 目次

- [概要](#概要)
- [DeepSeek R1とは](#deepseek-r1とは)
- [GRPO学習](#grpo学習)
- [SFT学習](#sft学習)
- [環境設定](#環境設定)
- [データ準備](#データ準備)
- [学習の実行](#学習の実行)
- [推論と評価](#推論と評価)
- [トラブルシューティング](#トラブルシューティング)

## 概要

DeepSeek R1は、数学的推論能力に特化した大規模言語モデルです。Colossal-AIでは、GRPO（Group Relative Policy Optimization）とSFT（Supervised Fine-Tuning）を使用してDeepSeek R1モデルの学習をサポートしています。

### 主な特徴

- **GRPO学習**: DeepSeek R1論文で使用された強化学習アルゴリズム
- **SFT学習**: 教師あり微調整による基礎的な能力向上
- **数学的推論**: 数学問題解決能力の向上
- **効率的な学習**: Colossal-AIの並列化技術による高速学習
- **検証可能な報酬**: 正確な答えに基づく報酬システム

## DeepSeek R1とは

DeepSeek R1は、DeepSeek V3をベースとして強化学習により数学的推論能力を向上させたモデルです。主な改善点：

- **思考プロセス**: `<think></think>`タグによる推論過程の明示
- **答え形式**: `<answer></answer>`タグによる最終答えの明確化
- **報酬システム**: 正確性に基づく段階的報酬

## GRPO学習

### GRPOとは

GRPO（Group Relative Policy Optimization）は、PPO（Proximal Policy Optimization）の変種で、以下の特徴があります：

- **グループベース学習**: 複数の応答を同時に生成・評価
- **相対的最適化**: グループ内での相対的な性能向上
- **メモリ効率**: PPOよりも効率的なメモリ使用

### 報酬関数

数学問題解決における報酬関数：

```python
def reward_function(response, ground_truth):
    # フォーマットチェック
    if not has_correct_format(response):
        return 0
    
    # 答えの正確性チェック
    if extract_answer(response) == ground_truth:
        return 10  # 正解
    else:
        return 1   # フォーマットは正しいが不正解
```

## SFT学習

### SFTとは

SFT（Supervised Fine-Tuning）は、教師ありデータを使用してモデルを微調整する手法です：

- **基礎能力向上**: モデルの基本的な理解能力を向上
- **フォーマット学習**: 適切な応答形式の学習
- **ドメイン適応**: 特定分野への適応

### SFT学習の実行

```bash
# SFT学習の基本コマンド
python applications/ColossalChat/examples/training_scripts/train_sft.py \
    --model_name_or_path deepseek-ai/DeepSeek-V3 \
    --dataset_name qwedsacf/competition_math \
    --output_dir ./output/deepseek-r1-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500
```

## 環境設定

### 必要な環境

- **Python**: 3.8以上
- **PyTorch**: 2.0以上
- **CUDA**: 11.8以上
- **GPU**: A100/H100推奨（最低でもV100）
- **メモリ**: 80GB以上のGPUメモリ推奨

### インストール

```bash
# Colossal-AIのインストール
pip install colossalai

# 追加の依存関係
pip install transformers datasets accelerate wandb

# DeepSeek用の追加パッケージ
pip install flash-attn --no-build-isolation
```

### モデルの準備

```bash
# DeepSeek V3モデルのダウンロード
huggingface-cli download deepseek-ai/DeepSeek-V3 --local-dir ./models/DeepSeek-V3
```

## データ準備

### データセットの選択

GRPO学習には以下のデータセットが推奨されます：

- **competition_math**: 数学競技問題
- **gsm8k**: 小学校レベルの数学問題
- **math**: 高校レベルの数学問題

### データセットの準備

```python
from datasets import load_dataset

# 数学競技データセットの読み込み
dataset = load_dataset("qwedsacf/competition_math")

# データの前処理
def preprocess_data(example):
    return {
        "prompt": example["problem"],
        "solution": example["solution"]
    }

dataset = dataset.map(preprocess_data)
```

### データ形式

学習データは以下の形式である必要があります：

```json
{
    "prompt": "次の方程式を解いてください: 2x + 3 = 7",
    "solution": "<think>2x + 3 = 7から、2x = 7 - 3 = 4、よってx = 2</think><answer>2</answer>"
}
```

## 学習の実行

### SFT学習（第1段階）

```bash
# 教師あり微調整
torchrun --nproc_per_node=8 applications/ColossalChat/examples/training_scripts/train_sft.py \
    --model_name_or_path ./models/DeepSeek-V3 \
    --dataset_name qwedsacf/competition_math \
    --output_dir ./output/deepseek-r1-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --report_to wandb
```

### GRPO学習（第2段階）

```bash
# GRPO強化学習
torchrun --nproc_per_node=8 applications/ColossalChat/examples/training_scripts/train_grpo.py \
    --model_name_or_path ./output/deepseek-r1-sft \
    --dataset_name qwedsacf/competition_math \
    --output_dir ./output/deepseek-r1-grpo \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --inference_batch_size 4 \
    --logits_forward_batch_size 1 \
    --initial_temperature 1.0 \
    --final_temperature 0.1 \
    --learning_rate 1e-6 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --report_to wandb
```

### パラメータの説明

- `num_generations`: 各プロンプトに対する生成数（通常8以上）
- `inference_batch_size`: 推論時のバッチサイズ
- `logits_forward_batch_size`: ロジット計算時のバッチサイズ
- `initial_temperature`: 初期温度（探索の多様性）
- `final_temperature`: 最終温度（収束時の安定性）

## 推論と評価

### 学習済みモデルの推論

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルとトークナイザーの読み込み
model_path = "./output/deepseek-r1-grpo"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 推論の実行
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

# 使用例
prompt = "次の方程式を解いてください: 3x - 5 = 16"
response = generate_response(prompt)
print(response)
```

### 評価スクリプト

```python
from datasets import load_dataset
import re

def extract_answer(text):
    """答えを抽出する関数"""
    match = re.search(r'<answer>(.*?)</answer>', text)
    return match.group(1) if match else None

def evaluate_model(model, tokenizer, dataset, num_samples=100):
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset[:num_samples]):
        prompt = example["problem"]
        ground_truth = example["answer"]
        
        response = generate_response(prompt)
        predicted_answer = extract_answer(response)
        
        if predicted_answer and predicted_answer.strip() == ground_truth.strip():
            correct += 1
        total += 1
        
        if i % 10 == 0:
            print(f"進捗: {i}/{num_samples}, 正解率: {correct/total:.2%}")
    
    accuracy = correct / total
    print(f"最終正解率: {accuracy:.2%}")
    return accuracy

# 評価の実行
test_dataset = load_dataset("qwedsacf/competition_math", split="test")
accuracy = evaluate_model(model, tokenizer, test_dataset)
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー

**問題**: `CUDA out of memory` エラーが発生する

**解決方法**:
```bash
# バッチサイズを削減
--per_device_train_batch_size 1
--gradient_accumulation_steps 16

# 推論バッチサイズを削減
--inference_batch_size 2
--logits_forward_batch_size 1

# 勾配チェックポイントを有効化
--gradient_checkpointing True
```

#### 2. 学習が収束しない

**問題**: 報酬が向上しない、または学習が不安定

**解決方法**:
```bash
# 学習率を調整
--learning_rate 5e-7

# 温度パラメータを調整
--initial_temperature 1.5
--final_temperature 0.05

# 生成数を増加
--num_generations 16
```

#### 3. 生成品質が低い

**問題**: モデルの出力が期待通りでない

**解決方法**:
- ベースモデルの品質を確認
- SFT（教師あり微調整）を事前に実行
- データセットの品質を確認
- 報酬関数の設計を見直し

### パフォーマンス最適化

#### メモリ使用量の最適化

```python
# モデル設定での最適化
model.config.use_cache = False
model.gradient_checkpointing_enable()

# データローダーの最適化
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

#### 学習速度の向上

```bash
# Flash Attentionの使用
pip install flash-attn --no-build-isolation

# コンパイル最適化
export TORCH_COMPILE=1

# 混合精度学習
--bf16 True
--tf32 True
```

## まとめ

このガイドでは、Colossal-AIを使用したDeepSeek R1モデルのSFTとGRPO学習について詳しく説明しました。主なポイント：

1. **環境設定**: 適切なハードウェアとソフトウェアの準備
2. **データ準備**: 数学問題データセットの前処理
3. **SFT学習**: 基礎的な能力向上のための教師あり学習
4. **GRPO学習**: 効率的な強化学習の実行
5. **推論と評価**: 学習済みモデルの性能評価
6. **トラブルシューティング**: よくある問題の解決方法

DeepSeek R1の数学的推論能力を最大限に活用するために、このガイドを参考にして学習を進めてください。

## 参考資料

- [DeepSeek R1論文](https://arxiv.org/abs/2501.12948)
- [Colossal-AI公式ドキュメント](https://colossalai.readthedocs.io/)
- [ColossalChatリポジトリ](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat)
- [GRPO実装詳細](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat/examples)

## サポート

質問や問題がある場合は、以下の方法でサポートを受けることができます：

- [GitHub Issues](https://github.com/hpcaitech/ColossalAI/issues)
- [GitHub Discussions](https://github.com/hpcaitech/ColossalAI/discussions)
- [Slack コミュニティ](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
