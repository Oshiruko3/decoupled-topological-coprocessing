# 並行位相幾何コプロセッシング (Decoupled Topological Coprocessing: DTC)
## 長文推論と生成における非侵襲的なハルシネーション抑止と軌道制御

[ **English** ](README.md) | [ **日本語** ](README_ja.md)

[![DOI](https://zenodo.org/badge/DOI/pending.svg)](https://doi.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper: PDF (EN)](https://img.shields.io/badge/Paper-PDF%20(English)-blue.svg)](paper/dtc_paper_en.pdf)
[![Paper: PDF (JA)](https://img.shields.io/badge/Paper-PDF%20(Japanese)-red.svg)](paper/dtc_paper_ja.pdf)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **著者**: 松本 幸太 (Kouta Matsumoto) - 独立研究者  
> **連絡先**: Oshiruko3@users.noreply.github.com

---

## 概要 (Overview)

**並行位相幾何コプロセッシング（Decoupled Topological Coprocessing: DTC）**は、大規模言語モデル（LLM）が複雑な多段階・長文推論を行う際に発生する「論理的堂々巡り」や「自己強化的な誤謬（ハルシネーション）」を実時間で検知・解決する、新しい非侵襲的システムアーキテクチャです。

従来の同期的な事後ガードレール（推論遅延の悪化）や、連続ベクトルステアリング（語彙崩壊・意味不明なトークン出力）に依存するのではなく、DTCは位相幾何学（トポロジー）監査レイヤーを本体生成エンジンから**直交的に分離（完全並行化）**します。

```text
[ 本体生成エンジン (LLM) ]
    トークン列: t_1 ---> t_2 ---> t_3 ---> t_4 ---> t_5 ---> (無停止の高速パス)
                                     ^
                                     | (異常確定時のみ非侵襲的に介入・切断)
[ 並行コプロセッサ (DTC) ]
    思考軌道ベクトル ---> [ スライディング窓 TDAスキャン (H1ループ検知) ]
                                     |
                             (認識的フィルタによる推移観測)
                                     v
                        [ 文脈的再アンカリング (Prefix Cachingロールバック + プロンプト) ]
```

---

## 主要な技術的革新 (Key Innovations)

1. **ハルシネーションの位相幾何学的定式化**:
   論理的循環や堂々巡りは、潜在空間上の思考軌道において「非自明な1次元パーシステントホモロジー閉曲線（$H_1$ サイクル）」として幾何学的に現れます。
2. **認識的状態遷移フィルタ（3状態の推移観測）**:
   従来のTDA監視が抱えていた「健全な検算や背理法の検証を誤遮断してしまう」という偽陽性（False Positive）問題を克服。0（正常）と1（異常）の二値判定ではなく、第3の状態**「State Both（保留監視）」**を導入。1〜2ステップの推移を見守ることで、健全な思考は自然に開いた螺旋軌道へと抜けて正常復帰し、悪質なループのみを確実に異常判定して遮断します。
3. **Prefix Caching連動による文脈的再アンカリング**:
   ループが確定した際、ループ直前の地点までトークン列を巻き戻し、内省トリガー（PreserveThinking Injection）を注入。CUDAカーネルを改変することなく、vLLM、SGLang、llama.cppなどの現代の推論エンジンに標準装備されている **Prefix Caching（KVキャッシュ再利用）** とシームレスに連動し、ゼロ遅延での思考経路の再分岐を実現します。
4. **極小のレイテンシオーバーヘッド (3.26%)**:
   軽量CPUスレッドでRipserスライディングウィンドウを非同期計算（1スキャン平均 **33.1ms**）。推論エンジンの生成スループットに対するオーバーヘッドはわずか **3.26%** にとどまります。

---

## 実証実験結果 (Gemma 4 26B on AIME/HLEベンチマーク)

外部検索が通用しない純粋な演繹・数学的推論課題（AIMEおよびHumanity's Last Examサブセット）において直接対決検証を実施：

| 評価指標 | ベースライン（素の推論） | DTC介入推論 | 改善効果 (Delta) |
| :--- | :---: | :---: | :---: |
| **完走率 (Answer Output Rate)** | **0.0%** (0 / 6問) | **83.3%** (5 / 6問) | **+83.3%** |
| **厳密正答率 (Exact Accuracy)** | **0.0%** (0 / 6問) | **33.3%** (2 / 6問) | **+33.3%** |
| **思考ステップ / トークン削減** | 基準 (52.0ステップで枯渇) | **17.3ステップで剪定** | **-66.7% (最大 -91.8%)** |
| **エンドツーエンド生成遅延** | 基準 (129.2 tok/s) | **125.0 tok/s** | **オーバーヘッドわずか 3.26%** |

![思考軌跡の2次元PCA射影](assets/cot_trajectory_pca.png)

---

## リポジトリ構成 (Repository Structure)

```text
decoupled-topological-coprocessing/
├── README.md               # 英語プロジェクト概要ドキュメント
├── README_ja.md            # 日本語プロジェクト概要ドキュメント (本書)
├── LICENSE                 # Apache License 2.0 (特許防衛条項付き)
├── CITATION.cff            # 学術引用メタデータ
├── paper/
│   ├── dtc_paper_en.pdf    # 英語版 論文PDF (出版フォーマット)
│   ├── dtc_paper_ja.pdf    # 日本語版 論文PDF (出版フォーマット)
│   ├── paper_en.md         # 英語版 ポジションペーパー本文
│   └── paper_ja.md         # 日本語版 ポジションペーパー本文
├── src/
│   ├── __init__.py         # パッケージエントリーポイント
│   ├── core.py             # TopologicalCoprocessor & EpistemicState (State Both)
│   ├── embeddings.py       # 思考埋め込みプロバイダー
│   ├── projector.py        # 思考軌道2次元PCAプロジェクター
│   └── dtc_live_monitor.py # リアルタイムSSEストリーミング監視＆自動遮断デモ
├── experiments/
│   ├── run_statistical_benchmark.py # AIME/HLEバッチベンチマーク実行スクリプト
│   ├── run_latency_overhead_benchmark.py # レイテンシ・スループット計測スクリプト
│   ├── aime_benchmark_batch_results.json # AIMEベンチマーク生データ
│   └── dtc_latency_overhead_benchmark.json # レイテンシ計測生データ
└── assets/
    └── cot_trajectory_pca.png # 思考軌道の2次元PCA比較プロット図
```

---

## クイックスタート (Quick Start)

### 1. 依存ライブラリのインストール
```bash
pip install requests numpy sentence-transformers ripser scikit-learn matplotlib
```

### 2. リアルタイム遮断デモの実行
ポート8000でOpenAI互換のローカル推論サーバー（llama-server など）が稼働していることを確認し、以下を実行します：
```bash
python src/dtc_live_monitor.py
```

### 3. ベンチマークとレイテンシ測定の再現
```bash
# AIME/HLE 難関推論バッチベンチマークを実行
python experiments/run_statistical_benchmark.py

# 実時間ストリーミングスループットのオーバーヘッドを測定
python experiments/run_latency_overhead_benchmark.py
```

---

## 学術引用 (Citation)

本ソフトウェア、データセット、または論文の知見を利用される際は、以下の形式で引用してください：

```bibtex
@article{matsumoto2026dtc,
  title={Decoupled Topological Coprocessing: Non-Intrusive Hallucination Mitigation and Trajectory Steering in Frontier Reasoning and Generation},
  author={Matsumoto, Kouta},
  year={2026}
}
```

---
*コード: Apache License 2.0. 論文およびドキュメント: CC-BY 4.0. 開発者: 松本 幸太 (Kouta Matsumoto).*