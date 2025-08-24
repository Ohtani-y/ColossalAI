<div align="center">
   <a href="https://www.colossalai.org/">
      <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/colossal-ai_logo_vertical.png" width="400" />
   </a>

   <div align="center">
      <a href="https://www.colossalai.org">ホームページ</a> |
      <a href="https://colossalai.readthedocs.io/en/latest/">ドキュメント</a> |
      <a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples">サンプル</a> |
      <a href="https://github.com/hpcaitech/ColossalAI/discussions">フォーラム</a> |
      <a href="https://medium.com/@hpcaitech">ブログ</a>
   </div>

   [![GitHub Repo stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)](https://github.com/hpcaitech/ColossalAI/stargazers)
   [![Build](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml/badge.svg)](https://github.com/hpcaitech/ColossalAI/actions/workflows/build_on_schedule.yml)
   [![Documentation](https://readthedocs.org/projects/colossalai/badge/?version=latest)](https://colossalai.readthedocs.io/en/latest/?badge=latest)
   [![CodeFactor](https://www.codefactor.io/repository/github/hpcaitech/colossalai/badge)](https://www.codefactor.io/repository/github/hpcaitech/colossalai)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/hpcai-tech)
   [![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w)
   [![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png)

   | [English](README.md) | [中文](README-zh-Hans.md) | [日本語](README_ja.md) |

</div>

## 最新ニュース
* [2024/12] [Colossal-AI 0.4.4リリース](https://github.com/hpcaitech/ColossalAI/releases/tag/v0.4.4): DeepSeek V3サポート、FP8通信、パフォーマンス最適化
* [2024/11] [Colossal-AI 0.4.3リリース](https://github.com/hpcaitech/ColossalAI/releases/tag/v0.4.3): Llama 3.2サポート、Mamba-2サポート、バグ修正
* [2024/09] [Colossal-AI 0.4.2リリース](https://github.com/hpcaitech/ColossalAI/releases/tag/v0.4.2): Qwen2.5サポート、Baichuan2サポート、パフォーマンス向上

## 目次

<details>
<summary>目次を表示</summary>

- [Colossal-AIとは](#colossal-aiとは)
- [機能](#機能)
- [実世界での応用](#実世界での応用)
  - [Colossal-LLaMA-2](#colossal-llama-2)
  - [ColossalChat](#colossalchat)
  - [AIGC](#aigc)
  - [生体医学](#生体医学)
- [並列学習デモ](#並列学習デモ)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [貢献方法](#貢献方法)
- [引用](#引用)

</details>

## Colossal-AIとは

[Colossal-AI](https://www.colossalai.org/)（旧ColossalAI）は、大規模AIモデルをより安価で高速、そしてアクセスしやすくするための統合深層学習システムです。効率的な並列化技術により、複数のGPUを持つ分散システムでのモデル学習を加速できます。また、単一GPUシステムでも動作します。

## 機能

Colossal-AIは、大規模モデルの学習と推論を効率化するための包括的な機能を提供します：

- **並列化戦略**: データ並列、テンソル並列、パイプライン並列、シーケンス並列、専門家混合（MoE）並列
- **異種メモリ管理**: CPU、GPU、NVMeメモリの効率的な利用
- **精度最適化**: 混合精度学習、FP16/BF16サポート、FP8通信
- **メモリ効率**: ZeRO、勾配チェックポイント、メモリオフロード
- **使いやすいツール**: Booster API、自動並列化、モデル分割
- **エコシステム統合**: HuggingFace Transformers、PyTorch Lightning、Ray完全対応

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## 実世界での応用

### Colossal-LLaMA-2

<div align="center">
   <a href="https://medium.com/@yangyou_berkeley/colossal-llama-2-7b-base-an-open-source-substitute-to-llama-2-pretraining-1f4b4f02a90b">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/colossal-llama-2/colossal-llama-2.png" width="700" />
   </a>
</div>

[Colossal-LLaMA-2](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA): LLaMA-2の継続事前学習により、わずか数千ドルで主流の大規模モデルと同等の結果を実現。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA)
[[ブログ]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFaceモデル]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)

### ColossalChat

<div align="center">
   <a href="https://www.youtube.com/watch?v=HcTiHzApHm0">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/applications/chat/ColossalChat%20YouTube.png" width="700" />
   </a>
</div>

[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat): 完全なRLHFパイプラインでChatGPTをクローンするオープンソースソリューション。DeepSeek R1のGRPO学習もサポート。
[[コード]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat)
[[ブログ]](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b)
[[デモ]](https://www.youtube.com/watch?v=HcTiHzApHm0)

- RLHF PPOステージ3学習で最大10倍高速化
- 単一サーバー学習で最大7.73倍、単一GPU推論で1.42倍高速化
- 単一GPUでのモデル容量を最大10.3倍拡大
- ミニデモ学習プロセスはわずか1.62GBのGPUメモリで実行可能

### AIGC
Stable Diffusion v1およびv2などのAIGC（AI生成コンテンツ）モデルの加速。

- **学習**: Stable Diffusionのメモリ消費を最大5.6倍削減、ハードウェアコストを最大46倍削減（A100からRTX3060へ）
- **DreamBooth微調整**: わずか3-5枚の画像でモデルをパーソナライズ
- **推論**: 推論時のGPUメモリ消費を2.5倍削減

### 生体医学
AlphaFoldタンパク質構造の加速

- **FastFold**: GPUクラスターでの学習と推論を加速、10000残基以上の推論シーケンス対応
- **Intel連携**: 3倍の推論加速と39%のコスト削減
- **xTrimoMultimer**: タンパク質モノマーとマルチマーの構造予測を11倍加速

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## 並列学習デモ

### LLaMA3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/examples/images/LLaMA3-70B-H100.png" width=600/>
</p>

- 700億パラメータのLLaMA3モデル学習を18%加速

### GPT-3
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT3-v5.png" width=700/>
</p>

- (GPT-3) 1750億パラメータモデルの学習を2-5倍加速

### PaLM
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM%20v2.png" width=700/>
</p>

- (PaLM) 5400億パラメータモデルの学習を2.5倍加速

<p align="right">(<a href="#top">トップに戻る</a>)</p>

## インストール

### PyPIからインストール

```bash
pip install colossalai
```

**注意**: PyPIで提供されるバージョンのみがサポートされています。夜間ビルドは提供していません。

### ソースからインストール

> このドキュメントは最新バージョン（mainブランチ）に対応しています。安定版をお探しの場合は、[最新リリース](https://github.com/hpcaitech/ColossalAI/releases)をご確認ください。

<details>
<summary><b>ソースからインストール</b></summary>

```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# CUDA拡張をインストール（オプション）
# CUDA_EXT=1 pip install -e .
pip install -e .
```

デフォルトでは、CUDA拡張なしでインストールされます。CUDA拡張を有効にするには、`CUDA_EXT=1`を設定してください。

</details>

## 使用方法

### セットアップなしで始める

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NuqXrqhTn32EB_SzOeaRJSp8b6LmJX3P?usp=sharing)

Colossal-AIを使用してResNetをCIFAR10データセットで学習する例：

```python
import torch
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from colossalai.nn.optimizer import HybridAdam
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import colossalai

# Colossal-AIを起動
colossalai.launch_from_torch()
coordinator = DistCoordinator()

# モデル、オプティマイザー、データセットを定義
model = resnet18(num_classes=10)
optimizer = HybridAdam(model.parameters(), lr=1e-3)
dataset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Boosterでモデルを強化
plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

# 学習ループ
model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        booster.backward(loss, optimizer)
        optimizer.step()
        
        if coordinator.is_master():
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
```

### DeepSeek R1との統合

Colossal-AIは、DeepSeek R1モデルの学習と推論を完全サポートしています。GRPO（Group Relative Policy Optimization）を使用した数学的推論能力の向上が可能です。

```python
# DeepSeek R1 GRPO学習の例
from colossalai.applications.colossalchat.examples.training_scripts import train_grpo

# GRPO学習設定
python train_grpo.py \
    --model_name_or_path deepseek-ai/DeepSeek-V3 \
    --dataset_name qwedsacf/competition_math \
    --num_generations 8 \
    --inference_batch_size 8 \
    --initial_temperature 1.0 \
    --final_temperature 0.1
```

詳細な使用方法については、[ColossalChatドキュメント](applications/ColossalChat/README.md)をご参照ください。

## 貢献方法

このプロジェクトは、コミュニティからの貢献を歓迎しています。

- バグ報告や機能リクエストは[Issues](https://github.com/hpcaitech/ColossalAI/issues)へ
- コード変更はPull Requestとして送信してください
- 詳細な貢献ガイドラインは[CONTRIBUTING_ja.md](CONTRIBUTING_ja.md)をご参照ください

プロジェクトを応援していただける場合は、ぜひスターを付けてください！

## 引用

```bibtex
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```

Colossal-AIは[HPC-AI Tech](https://hpc-ai.tech/)によって開発されています。
