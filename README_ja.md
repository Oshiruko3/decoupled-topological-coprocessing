# 非侵襲並行位相幾何コプロセッシング（DTC v3.0）
## 中間フォーク力学系と決定論的認知状態ガバナンスによる極小遅延推論安定化

[ **English** ](README.md) | [ **日本語** ](README_ja.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22726133.svg)](https://doi.org/10.5281/zenodo.22726133)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DTC v3.0 論文: PDF (日本語版)](https://img.shields.io/badge/DTC%20v3.0%20論文-PDF%20(JA)-red.svg)](paper/v3/dtc_v3_paper_ja.pdf)
[![DTC v3.0 Paper: PDF (英語版)](https://img.shields.io/badge/DTC%20v3.0%20Paper-PDF%20(EN)-blue.svg)](paper/v3/dtc_v3_paper_en.pdf)
[![DTC v2.0 アーカイブ](https://img.shields.io/badge/DTC%20v2.0-Archive-gray.svg)](paper/v2/)
[![DTS ポジションペーパー](https://img.shields.io/badge/DTS%20Paper-Position%20Paper-purple.svg)](paper/dts/)

> **著者**: 松本 幸太（Kouta Matsumoto / 独立研究者）  
> 連絡先: `Oshiruko3@users.noreply.github.com`  
> 公式 DOI: `10.5281/zenodo.22726133`（v2.0 アーカイブ / v3.0 正式公開版）

---

## 概要

**非侵襲並行位相幾何コプロセッシング（Decoupled Topological Coprocessing; DTC v3.0）** は、推論時計算（Test-Time Compute）を拡張した大規模推論モデル（LRMs）において発生する推論の膠着（循環論法）や思考崩壊（同一文字列の反復・カタトニックロック）を極小遅延かつ非侵襲に統御する並行ソフトウェアランタイムです。

先行研究（DTC v2.0）では持続的ホモロジー（1次ベッチ数 $H_1$）を用いたループやハルシネーションの二値検知を確立しましたが、推論軌道の運動力学的な速度変化や、長考・知的跳躍といった多様な認知状態をきめ細かく統御する枠組みには至っていませんでした。

DTC v3.0 では、ヴィートリス・リップス複体簡約（Ripser）の中間生成物である距離行列 $\mathbf{D} \in \mathbb{R}^{N \times N}$ を分岐利用する**運動力学的中間フォーク**を導入。追加の距離計算を一切伴わずに、回転半径 $R_g$、終端速度 $v$、終端加速度 $\Delta v$、局所最大リアプノフ指数 $\lambda_{\max}$ を $0.13\ \mathrm{ms}$ で並行抽出します。これらを**位相的認知決定マトリクス（P1〜P7）**と統合することで、**平均 0.5858 ms（P99 < 1.0 ms）の極小遅延**で推論プロセス全体の包括的ガバナンスを実現しました。

```text
[ 主推論エンジン (Host LLM Engine) ]
       │ （テキストトークンストリーム: SSE / WebSocket）
       ▼
[ ステージ 1: 命題節セグメンタ (Clause Segmenter) ]
       │ （句読点・改行・記号による意味境界分割 / 文長 L ≥ 12）
       ▼
[ ステージ 2: 軽量埋め込みプロセッサ (Embedding Engine) ]
       │ （all-MiniLM-L6-v2, d=384, FP16: 768 bytes/step）
       ▼
[ ステージ 3: DTC並行演算カーネル (DTC Coprocessor Kernel) ] (0.58 ms)
       ├── 距離行列 D ∈ R^{N×N} の構築 (0.03 ms, N=16)
       ├── 運動力学的中間フォーク (Rg, v_term, Δv_term, λ_max) (0.13 ms)
       └── Vietoris-Rips 複体簡約 (Ripser: H1 持続ホモロジー) (0.38 ms)
       │
       ▼
[ ステージ 4: 位相的認知決定マトリクス (Decision Matrix & Dispatcher) ]
       │ （P1–P7 分類 ＆ ヒステリシス持続監視: Streak ≥ 2）
       ▼
[ 制御シグナル (PASS / OBSERVE / ABNORMAL_TERMINATE) ] ──▶ 主推論エンジン
```

---

## DTC v3.0 の主要な技術的革新

1. **運動力学的中間フォーク（0.13 ms の並行抽出）**:
   持続ホモロジー簡約の入力として作られる距離行列 $\mathbf{D}$ に、軌道の運動力学情報が完全に内包されていることを定式化。$\mathbf{D}$ から直接、回転半径 $R_g$、終端速度 $v$、加速度 $\Delta v$、局所最大リアプノフ指数 $\lambda_{\max}$ を抽出することで、重複する距離計算をゼロに抑えました。
2. **サブミリ秒遅延と極小バス帯域消費**:
   アルゴリズム全体の平均処理遅延は **0.5858 ms**（99パーセンタイル **0.9744 ms**）を記録。命題節の生成間隔（$80\sim 120\ \mathrm{ms}$）に対して 0.5% 未満のオーバーヘッドであり、体感生成速度を一切低下させません。また、必要なデータ転送レートは毎秒 **7.68 KB** であり、最新の AXI-4 や PCIe 帯域の **0.0001% 未満** で動作します。
3. **位相的認知決定マトリクス（P1〜P7 分類体系）**:
   従来の「正常 / 異常」の二値検知を超え、推論状態を7つの位相パターンへ決定論的に分類・調律します。
   * **P1 (Grounded)**: 健全かつ直線的な論理演繹 $\to$ `PASS_THROUGH`（無介入通過）
   * **P2 (Watchlist)**: 思考の逡巡、一時的な反復 $\to$ `OBSERVE`（監視バッファリング）
   * **P3 (Deadlock)**: 1次循環トラップ（堂々巡り） $\to$ `MINIMAL_ANCHOR`（巻き戻し脱出）
   * **P4 (Slip)**: 計算符号ミスや事実誤認 $\to$ `PASS_THROUGH`（不干渉・ポリシーA）
   * **P5 (Delusion)**: 孤立した捏造や前提の弱い幻覚 $\to$ `UNIVERSAL_ANCHOR`（普遍接地）
   * **P6 (Collapse)**: カタトニック・ロック、思考熱死 $\to$ `ABNORMAL_TERMINATE`（ハードキル）
   * **P7 (Creative Leap)**: 異分野へのアナロジー展開 $\to$ `PASS_THROUGH`（知的跳躍の保護）
4. **思考解像度スケーリングの発見とアライメント税ゼロ**:
   8,192トークンの無制限ストリーム環境において、26Bモデルは数学証明（S1）で最大 **94ステップ** に及ぶ多角的自己反証思考を展開しました。DTCは高い回転半径（$R_g = 0.751$）と正のリアプノフ指数（$\lambda_{\max} = +0.042$）を捉え、**誤介入率 0.0% で94ステップの完走を完全保護**しました。
5. **98.2% のトークン・エネルギー削減（ROI）**:
   循環トラップ（S2）および思考崩壊（S4）に対しては、わずか **10〜13ステップ（1.2〜2.0秒）** で異常軌道を特定して即時遮断を執行。1リクエストあたり **8,000トークン以上（98.0%〜98.4%）の計算資源とGPU電力を削減**しました。
6. **量子化ノイズ補償仮説の実証**:
   4bit量子化モデル（26B Q4）において丸め誤差の離散ポテンシャル面に捕獲されて即座に思考崩壊（終端速度 $v_{\mathrm{term}} \to 0$）に至る挙動と、完全精度モデル（E2B FP16）が滑らかな勾配により思考を維持する挙動を直接対比。DTCが低ビット運用の物理的脆弱性を外部からリアルタイム補償することを実証しました。

---

## 実機検証結果（8k無制限コンテキスト対照実験）

最大 8,192 トークンの無制限コンテキスト下で、**Gemma 4 26B (4-bit Q4_0)** と **Gemma 4 E2B (FP16非量子化)** に対する厳密な実機対照実験（$N=3$ 独立試行、計30試行、総計989ステップ）を実施しました。

### シナリオ別実証結果

| シナリオ | モデル | 平均評価ステップ | 平均時間 (s) | 平均消費トークン | 判定結果 | 執行アクション | トークン削減率 (ROI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1: 素数証明** | 26B Q4 | 63.0 (最大94) | 10.75 | 1,311.3 | `P1_GROUNDED` | PASS_THROUGH | **誤介入 0.0%（完走）** |
| | E2B FP16 | 54.0 (最大78) | 15.92 | 1,111.3 | `P1_GROUNDED` | PASS_THROUGH | **誤介入 0.0%（完走）** |
| **S2: 循環ループ** | 26B Q4 | **12.3** | **1.40** | **165.3** | `P3_DEADLOCK` | MINIMAL_ANCHOR | **98.0% 削減** (~8,027 tok) |
| | E2B FP16 | **10.0** | **2.00** | **132.7** | `P3_DEADLOCK` | MINIMAL_ANCHOR | **98.4% 削減** (~8,059 tok) |
| **S3: 事実スリップ** | 26B Q4 | 45.7 | 9.19 | 1,135.3 | `P1 / P7` | PASS_THROUGH | **誤介入 0.0%（完走）** |
| | E2B FP16 | 41.0 | 14.04 | 973.3 | `P1_GROUNDED` | PASS_THROUGH | **誤介入 0.0%（完走）** |
| **S4: 思考崩壊** | 26B Q4 | **11.0** | **1.26** | **149.0** | `P6_COLLAPSE` | ABNORMAL_TERM | **98.2% 削減** (~8,043 tok) |
| | E2B FP16 | 12.0 | 7.14 | 495.0 | `P7` (メタ思考) | PASS_THROUGH | 正常思考継続 |
| **S5: 創造的跳躍** | 26B Q4 | 49.3 | 9.28 | 1,142.3 | `P1 / P7` | PASS_THROUGH | **誤介入 0.0%（完走）** |
| | E2B FP16 | 61.3 | 20.39 | 1,421.3 | `P1 / P7` | PASS_THROUGH | **誤介入 0.0%（完走）** |

### アルゴリズム処理遅延の実測値（計989ステップ）

| 評価指標 | Gemma 4 26B Q4 (499 steps) | Gemma 4 E2B FP16 (490 steps) |
| :--- | :---: | :---: |
| **平均遅延 (Mean)** | **0.5858 ms** | **0.5488 ms** |
| **中央値 (Median, P50)** | **0.6000 ms** | **0.5500 ms** |
| **95パーセンタイル (P95)** | **0.7381 ms** | **0.6946 ms** |
| **99パーセンタイル (P99)** | **0.9744 ms** | **0.7519 ms** |
| **最大遅延 (Max)** | 1.0850 ms | 0.7990 ms |
| **標準偏差 (Std)** | 0.1155 ms | 0.0963 ms |

---

## リポジトリ構成

```text
decoupled-topological-coprocessing/
├── benchmarks/
│   ├── datasets/                  # AIME 2026 および Trident 評価データセット
│   └── results/
│       ├── v2/                    # v2.0 評価トレース（AIME 2026, Trident, アブレーション）
│       └── v3/                    # ★ v3.0 実機対照実験ログ（8kコンテキスト、989ステップ）
│           ├── dtc_v3_validation_results.json
│           ├── dtc_v3_tri_replicate_results.json
│           ├── dtc_v3_e2b_benchmark_results.json
│           └── dtc_v3_empirical_summary.json
│
├── paper/
│   ├── compile_pdf.py             # 自動LaTeXビルドスクリプト（v3 / v2 両対応）
│   ├── v1/                        # 初代 DTC 論文アーカイブ
│   ├── v2/                        # DTC v2.0 正式論文アーカイブ（Zenodo DOI）
│   ├── v3/                        # ★ DTC v3.0 決定版論文・TeXソース・原稿
│   │   ├── dtc_v3_paper_ja.pdf    # 日本語版論文（PDF、9ページ）
│   │   ├── dtc_v3_paper_ja.tex    # 日本語版 LaTeX ソース（LuaLaTeX-Ja）
│   │   ├── dtc_v3_paper_ja_draft.md # 日本語版ドラフト原稿
│   │   ├── dtc_v3_paper_en.pdf    # 英語版論文（PDF、9ページ）
│   │   ├── dtc_v3_paper_en.tex    # 英語版 LaTeX ソース（pdfLaTeX）
│   │   └── dtc_v3_paper_en_draft.md # 英語版ドラフト原稿
│   └── dts/                       # 独立型セーフガード（DTS）ポジションペーパー
│
├── scripts/                       # 実機ベンチマーク実行および分析スクリプト群
│   ├── v1/                        # DTC v1.0 初期遅延・統計実験コード
│   │   ├── run_latency_overhead_benchmark.py
│   │   └── run_statistical_benchmark.py
│   └── v3/                        # ★ DTC v3.0 実機対照実験・分析コード
│       ├── benchmark_dtc_v3_tri_replicate.py
│       ├── benchmark_dtc_v3_e2b.py
│       ├── validate_dtc_v3_live.py
│       └── analyze_benchmarks.py
│
├── src/                           # コア実装
│   ├── shadow_core.py             # ★ DTC v3.0 中間フォーク・コプロセッサ
│   ├── decision_matrix.py         # ★ 位相的認知決定マトリクス（P1〜P7）
│   ├── core.py                    # DTC v1/v2 リファレンス実装
│   ├── embeddings.py              # MiniLM 埋め込みラッパー
│   └── projector.py               # 多様体射影ユーティリティ
│
├── CITATION.cff                   # 引用メタデータ
├── LICENSE                        # Apache 2.0 ライセンス
└── README_ja.md                   # 日本語ドキュメント
```

---

## クイックスタート

### インストール

```bash
git clone https://github.com/Oshiruko3/decoupled-topological-coprocessing.git
cd decoupled-topological-coprocessing

pip install numpy requests sentence-transformers ripser
```

### Python API の利用例

```python
from src.shadow_core import ShadowTopologicalCoprocessor
from src.embeddings import MiniLMEmbeddingProvider
from src.decision_matrix import diagnose_cognitive_state, DecisionAction

# 埋め込みプロバイダと DTC v3 コプロセッサ（ウィンドウ長 N=16）を初期化
embedder = MiniLMEmbeddingProvider()
coprocessor = ShadowTopologicalCoprocessor(window_size=16)

# 推論ストリームの命題節を順次処理
clauses = [
    "素数が有限個しか存在しないと背理法のために仮定する。",
    "すべての素数の集合を P = {p_1, p_2, ..., p_n} とする。",
    "ここで整数 N = p_1 * p_2 * ... * p_n + 1 を構成する。",
    # ... 後続の推論節
]

for clause in clauses:
    vector = embedder.embed(clause)
    metrics = coprocessor.step(vector)
    
    if metrics:
        diagnosis = diagnose_cognitive_state(metrics)
        print(f"節: '{clause[:30]}...' -> パターン: {diagnosis.pattern.name}, アクション: {diagnosis.action.name}")
        
        if diagnosis.action == DecisionAction.ABNORMAL_TERMINATE:
            print("思考崩壊を検知しました。ストリームを安全に遮断します。")
            break
```

---

## 引用 (Citation)

本研究成果、中間フォーク・パラダイム、または実験データを学術研究等で引用される場合は、以下の BibTeX をご利用ください。

```bibtex
@article{matsumoto2026dtc_v3,
  author    = {Matsumoto, Kouta},
  title     = {Decoupled Topological Coprocessing (DTC v3.0): Sub-Millisecond Intermediate Fork Dynamics and Deterministic Cognitive State Governance for Quantized Reasoning Models},
  year      = {2026},
  month     = {September},
  url       = {https://github.com/Oshiruko3/decoupled-topological-coprocessing}
}

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

本プロジェクトは **Apache License 2.0** の下で公開されています。詳細は [LICENSE](LICENSE) をご参照ください。