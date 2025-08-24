# クイックデモ

Colossal-AIは、効率的な並列化技術を備えた統合大規模深層学習システムです。このシステムは、並列化技術を適用することで、複数のGPUを持つ分散システムでのモデル学習を加速できます。また、単一GPUのシステムでも動作します。以下に、Colossal-AIの使用方法を示すクイックデモを紹介します。

## 単一GPU

Colossal-AIは、単一GPUのシステムで深層学習モデルを学習し、ベースライン性能を達成するために使用できます。単一GPUで[CIFAR10データセットでResNetを学習する例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet)を提供しています。この例は[ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples)で見つけることができます。詳細な手順は、その`README.md`に記載されています。

## 複数GPU

Colossal-AIは、複数のGPUを持つ分散システムで深層学習モデルを学習し、効率的な並列化技術を適用することで学習プロセスを大幅に加速するために使用できます。いくつかの並列化手法を試すことができます。

#### 1. データ並列

上記の単一GPUデモと同じ[ResNet例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet)を使用できます。`--nproc_per_node`をマシンのGPU数に設定することで、この例はデータ並列の例になります。

#### 2. ハイブリッド並列

ハイブリッド並列には、データ、テンソル、パイプライン並列が含まれます。Colossal-AIでは、異なるタイプのテンソル並列（1D、2D、2.5D、3D）をサポートしています。`config.py`の設定を変更するだけで、異なるテンソル並列を切り替えることができます。[GPT例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt)に従うことができます。詳細な手順は、その`README.md`に記載されています。

#### 3. MoE並列

MoE並列を実証するために[ViT-MoEの例](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/moe)を提供しています。WideNetは、より良い性能を達成するために専門家混合（MoE）を使用します。詳細は[チュートリアル: 専門家混合をモデルに統合する](../advanced_tutorials/integrate_mixture_of_experts_into_your_model.md)で確認できます。

#### 4. シーケンス並列

シーケンス並列は、NLPタスクにおけるメモリ効率とシーケンス長制限の問題に対処するために設計されています。[ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples)で[BERTの例](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/sequence_parallel)を提供しています。`README.md`に従ってコードを実行できます。

#### 5. DeepSeek R1 GRPO学習

Colossal-AIは、DeepSeek R1モデルのGRPO（Group Relative Policy Optimization）学習をサポートしています。数学的推論能力の向上に特化した学習が可能です。

```bash
# DeepSeek R1 GRPO学習の例
python applications/ColossalChat/examples/training_scripts/train_grpo.py \
    --model_name_or_path deepseek-ai/DeepSeek-V3 \
    --dataset_name qwedsacf/competition_math \
    --num_generations 8 \
    --inference_batch_size 8 \
    --initial_temperature 1.0 \
    --final_temperature 0.1
```

詳細については、[ColossalChatドキュメント](../../applications/ColossalChat/README.md)をご参照ください。

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 run_demo.py  -->
