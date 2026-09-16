# 並行位相幾何コプロセッシング（DTC v2.0）
## 大規模言語モデルにおける推論膠着の抑止と非侵襲的軌道調律

[ **English** ](README.md) | [ **日本語** ](README_ja.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726133.svg)](https://doi.org/10.5281/zenodo.22726133)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DTC v2.0 論文: PDF (日本語版)](https://img.shields.io/badge/DTC%20v2.0%20論文-PDF%20(JA)-red.svg)](paper/dtc_v2_paper_ja.pdf)
[![DTC v2.0 論文: PDF (英語版)](https://img.shields.io/badge/DTC%20v2.0%20Paper-PDF%20(EN)-blue.svg)](paper/dtc_v2_paper_en.pdf)
[![DTS ポジションペーパー: PDF (日本語版)](https://img.shields.io/badge/DTS%20提案論文-PDF%20(JA)-orange.svg)](paper/dts_position_paper_ja.pdf)
[![DTS Position Paper: PDF (英語版)](https://img.shields.io/badge/DTS%20Paper-PDF%20(EN)-purple.svg)](paper/dts_position_paper_en.pdf)

> **著者**: 松本 幸太（Kouta Matsumoto / 独立研究者）  
> 連絡先: `Oshiruko3@users.noreply.github.com`  
> 公式 DOI: `10.5281/zenodo.22726133`

---

## 概要

**並行位相幾何コプロセッシング（Decoupled Topological Coprocessing; DTC v2.0）** は、長考型推論モデル（LRMs）が直面する推論の循環固着（Circular Reasoning）や文脈アンカーを喪失した外因的ハルシネーション（Extrinsic Hallucination）を実時間で検知・解消する、非侵襲・非同期型のコプロセッサ・アーキテクチャです。

従来の事後強化学習（RLHF）や直列配置のガードレール判定器が抱える生成遅延の増大やアライメント税（知能の低下）を回避するため、DTCは**推論エンジンと完全に物理的・論理的に独立した並行監視パイプライン**を採用しています。Server-Sent Events（SSE）経由で流出するトークン列のスライディングウィンドウ（$W=8$ ステップ）に対し、1次元持続的ホモロジー（$H_1$ サイクル）を平均 **38.61 ms**（トークン生成間隔内に完全隠蔽）で計算し、異常な幾何学的閉路（アトラクター）を実時間で捕捉します。

```text
[ 主推論プロセッサ (LLM) ] ──(SSEストリーム)──▶ [ 命題バッファ分割 ] (L ≥ 12)
                                                        │
                                                        ▼
                                                 [ 時間局所スライディング窓 W=8 ]
                                                        │
                                                        ▼
                                                 [ 文脈埋め込みモデル ] (d=384)
                                                        │
                                                        ▼
                                                 [ Ripser エンジン ] (H1 持続性計算)
                                                        │
                 ┌──────────────────────────────────────┴──────────────────────────────────────┐
                 ▼                                                                             ▼
     正常判定: 監視継続 (ρ < τ)                                                     異常検知 (ρ ≥ 0.12)
                                                                                               │
                                                                                               ▼
                                                                                💥 [ 低遅延遮断 ] (TCP切断)
                                                                                               │
                                                                                               ▼
                                                                                [ KVキャッシュ巻き戻し ＆ 内省アンカー注入 ]
```

---

## DTC v2.0 の主要な工学的革新

1. **非侵襲的コプロセッシングとアライメント・タックス・ゼロ**:
   モデル重みや計算カーネルを一切改変せず、APIプロキシ層でのみ動作します。米国数学招待試験（**AIME 2026 公式問題 15問, $N=3$**）において、DTCは**誤介入率 0.0\%（100\%完全通過）**を達成し、健全な数学的演繹や背理法を一切阻害しないことを実証しました。
2. **低遅延遮断と Prefix Caching 連動ロールバック**:
   異常検知時に通信ソケットを切断して生成スレッドをミリ秒単位で停止。その後、循環突入前の安全点 $t_{\mathrm{rollback}} = t_{\mathrm{interrupt}} - k$（$k=2$）までテキストを巻き戻し、不変のメタ認知的内省アンカーを付加して再送します。Prefix Cachingによりプレフィル再計算コストは**実質ゼロ（0 ms）**となり、注意機構を開かれた演繹多様体へと不連続に相転移（Phase Transition）させます。
3. **敵対的 Trident ベンチマークによる実証**:
   循環論法トラップ（5問）および架空知識捏造トラップ（5問）の計10問（全30試行）において、**93.3\%（28/30試行）の早期異常遮断**と**100\%の膠着脱出・正常収束**を達成。総計20,705トークン（45.3\%減）を削減し、監視コストに対する**純エネルギー削減比（ROI）は 99.2\%**に達しました。
4. **量子化ノイズ補償仮説（Quantization Noise Compensation Hypothesis）の実証**:
   完全精度モデル（2B FP16）と4bit量子化モデル（26B Q4_0 + 4bit KV）の直接対照実験により、4bitの丸めノイズが自己注意機構に擬似アトラクターを形成して思考を不可逆的に固着させる物理的機序（自力脱出 0.0\%）を特定。滑らかな勾配を保持するFP16における自然終息現象（23.3\%）との比較を通じ、DTCがエッジ・低ビット運用の脆弱性を外付けで克服する不可欠な軌道調律機構であることを証明しました。
5. **線形計算量とフェイルオープン保証**:
   2次元単体数は $\binom{8}{3} = 56$ 個に厳格に上界されており、計算スパイクは幾何学的に排除されています。また、高負荷時に計算が遅延しても主推論を待機させず次窓へ監視を持ち越すフェイルオープン設計により、システムの可用性を100\%維持します。

---

## 実証実験データ

### 1. 数学演繹における健全性検証（AIME 2026 公式問題）
*評価モデル: Gemma 4 E2B (AMD ROCm環境, FP16)*

| 評価条件 | 正解数 (Mean $\pm$ Std) | 正答率 | 誤介入率 | 挙動特性 |
| :--- | :---: | :---: | :---: | :--- |
| **同期ベースライン** (`stream: False`) | $6.0 \pm 0.0$ / 15 | 40.0% | N/A | HTTPバッファ圧迫による早期途絶 |
| **純ストリーミング** (`stream: True`) | $6.0 \pm 0.0$ / 15 | 40.0% | N/A | 長大思考完走（$<9.5$k tok）、確率分散 |
| **DTC v2.0 (提案手法)** | $\mathbf{8.0 \pm 0.82}$ \textbf{/ 15} | $\mathbf{53.3\%}$ | $\mathbf{0.0\%}$ **(100% 通過)** | **健全な演繹を一切阻害せず完全通過** |

### 2. Trident 評価スイートにおける能動的防護性能（$N=3$, 全30試行）
*評価モデル: Gemma 4 26B (CUDA環境, Q4_0 / 128k ctx)*

| 問題ID | 分類 | 対象プロンプトの論理的焦点 | ベースライン消費 | DTC介入率 | DTC平均トークン |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **LOOP_01** | 循環トラップ | 3者循環の身代わり論法 | 1,024t (膠着) | **3/3 (100%)** | $398.3 \pm 42.1$ |
| **LOOP_02** | 循環トラップ | 不静定モジュラー連立方程式 ($2x+2y=21$) | 1,024t (膠着) | **3/3 (100%)** | $408.0 \pm 18.5$ |
| **LOOP_03** | 循環トラップ | 時間ループの自己因果パラドックス | 1,024t (膠着) | **3/3 (100%)** | $402.7 \pm 38.6$ |
| **LOOP_04** | 循環トラップ | 同語反復代数ループ ($x^2 - 4 = (x-2)(x+2)$) | 1,024t (膠着) | **3/3 (100%)** | $373.0 \pm 46.1$ |
| **LOOP_05** | 循環トラップ | 床屋のパラドックス（二値振動） | 1,024t (膠着) | **3/3 (100%)** | $368.0 \pm 23.3$ |
| **HALU_01** | 幻覚トラップ | 架空の2021年Nature論文引用要求 | 1,024t (膠着) | **3/3 (100%)** | $501.0 \pm 25.1$ |
| **HALU_02** | 幻覚トラップ | 1984年リヒテンシュタイン・アンドラ海戦 | 1,024t (膠着) | **2/3 (67%)** | $741.0 \pm 245.5$ |
| **HALU_03** | 幻覚トラップ | 光合成を行うアデリーペンギン（架空生物） | 1,024t (膠着) | **3/3 (100%)** | $471.7 \pm 31.9$ |
| **HALU_04** | 幻覚トラップ | 2011年ネオ・ビザンチン合意プロトコル | 1,024t (膠着) | **3/3 (100%)** | $488.7 \pm 41.2$ |
| **HALU_05** | 幻覚トラップ | 1783年京都条約（架空の講和条約） | 1,020t (膠着) | **2/3 (67%)** | $807.0 \pm 153.4$ |

### 3. モデル規模および精度横断対比（26B Q4_0 vs 2B FP16）

| 評価指標 | Gemma 4 26B (Q4_0 + 4bit KV) | Gemma 4 E2B (FP16 完全精度) | 数理的・物理的解釈 |
| :--- | :---: | :---: | :--- |
| **全体介入発火率** | **93.3%** (28/30試行) | **76.7%** (23/30試行) | FP16は自力収束の余地を保持 |
| **循環思考（LOOP）介入率** | **100.0%** (15/15試行) | **86.7%** (13/15試行) | Q4_0は100%捕捉、FP16は自力脱出あり |
| **幻覚誘導（HALU）介入率** | **86.7%** (13/15試行) | **66.7%** (10/15試行) | 早期慎重回答による通過（Pass） |
| **自然終息率（自力脱出）** | **0.0%** (0/30試行) | **23.3%** (7/30試行) | **滑らかなFP16勾配 vs 量子化ノイズトラップ** |
| **介入後循環残存 ($H_1$)** | **0.0** (完全根絶) | **0.0** (完全根絶) | 介入後は全問で100%正常収束 |
| **総トークン削減量** | **20,705 トークン** (45.3%減) | **4,390 トークン** (16.0%減) | 非生産的トークン消費の大幅抑制 |

---

## リポジトリ構成

```text
.
├── benchmarks/
│   ├── datasets/
│   │   ├── trident_benchmark_suite.json    # Trident 10問（LOOP 5問 + HALU 5問）
│   │   ├── aime_2026_15problems.json       # AIME 2026 公式問題 15問
│   │   └── aime_2026_full_30problems.json  # AIME 2026 全30問データセット
│   ├── results/
│   │   ├── aime_2026/                      # AIME 2026 実行ログ
│   │   └── trident/                        # Trident 実行ログ（26B & 2B & アブレーション）
│   ├── runner_sample.py                    # スタンドアロン追試実行スクリプト
│   └── README.md                           # ベンチマーク仕様書
│
├── paper/
│   ├── dtc_v2_paper_ja.pdf                 # DTC v2.0 論文（日本語版 PDF）
│   ├── dtc_v2_paper_en.pdf                 # DTC v2.0 論文（英語版 PDF）
│   ├── dtc_v2_paper_ja.tex                 # 日本語 LaTeX ソース
│   ├── dtc_v2_paper_en.tex                 # 英語 LaTeX ソース
│   ├── dtc_v2_paper_ja.md                  # Markdown 原稿（日本語）
│   ├── dtc_v2_paper_en.md                  # Markdown 原稿（英語）
│   ├── compile_pdf.py                      # 自動組版スクリプト
│   ├── dtc_paper_en.pdf                    # DTC v1.0 アーカイブ (英語)
│   ├── dtc_paper_ja.pdf                    # DTC v1.0 アーカイブ (日本語)
│   ├── dts_position_paper_en.pdf           # DTS ポジションペーパー (英語)
│   └── dts_position_paper_ja.pdf           # DTS ポジションペーパー (日本語)
│
├── src/                                    # DTC v1.0 リファレンス実装
├── CITATION.cff                            # 引用文献情報 (v2.0.0)
├── LICENSE                                 # Apache 2.0 ライセンス
└── README_ja.md                            # 本ドキュメント
```

---

## 再現手順（クイックスタート）

OpenAI互換の推論エンドポイントに対して評価を実行する手順です：

```bash
# リポジトリのクローン
git clone https://github.com/Oshiruko3/decoupled-topological-coprocessing.git
cd decoupled-topological-coprocessing

# 依存パッケージのインストール
pip install requests sentence-transformers ripser numpy

# Trident ベンチマークの実行（DTC有効モード）
python benchmarks/runner_sample.py \
  --api-base http://localhost:8000/v1 \
  --model gemma-4-26b \
  --dataset benchmarks/datasets/trident_benchmark_suite.json \
  --mode dtc \
  --output benchmarks/results/reproduction_run.json
```

---

## 引用 (Citation)

本研究の成果、ベンチマークデータセット、またはアーキテクチャをご利用の際は、以下のフォーマットで引用してください：

```bibtex
@article{matsumoto2026dtc_v2,
  author    = {Matsumoto, Kouta},
  title     = {Decoupled Topological Coprocessing for Mitigating Reasoning Deadlocks and Non-Invasive Trajectory Steering in Large Language Models (DTC v2.0)},
  year      = {2026},
  month     = {September},
  doi       = {10.5281/zenodo.22726133},
  url       = {https://github.com/Oshiruko3/decoupled-topological-coprocessing}
}
```

---

## ライセンス

本プロジェクトは **Apache License 2.0** のもとで公開されています。詳細は [LICENSE](LICENSE) をご参照ください。