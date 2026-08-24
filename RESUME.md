# RESUME — 続きをやる人へ

進んだところと、決めたことの理由。**測った数だけ書く**（「速いはず」は書かない）。

## いまどこ

| | 状態 |
|---|---|
| C++ 側: データ生成 → 学習 → 保存 → 推論 → 評価 | 通っている |
| Python 側: 学習 → 保存 → 推論 → 評価 | 通っている |
| 両言語のパリティ（ロジット・損失・**重み 1 本ずつの勾配**・生成） | PASS |
| 重みの受け渡し（C++ ⇄ Python、同じ `.pt`） | 両方向 PASS |
| GPU 学習（Kaggle T4、kbridge 経由） | 30000 step = 3388 秒（13.7 ms/サンプル） |
| 見たことのない式 2000 件での完全一致 | **1975 / 2000（98.8%）** |
| 本家 pix2tex の学習済みモデルを両言語で動かす | PASS（本家と 1 文字も違わない） |
| WASM デモ（wasm/） | 40 枚とも native と同じ読み・正解と一致 |

最後に通したもの:

```
selftest              往復 500/500、勾配の方向微分 最悪 3.71e-03  PASS
parity/model.py       ロジット 2.95e-07 / 損失 0 / 勾配 2.98e-06 / 生成一致  PASS
parity/interchange.py C++ が書いた .pt 4/4、python が書いた .pt 4/4  PASS
parity/pix2tex.py     試験画像 6 枚すべて C++ = python  PASS
wasm/test_node.js     40 枚: 正解と 40/40、native と 40/40  ok
```

公開先は <https://yomei-o.github.io/latex_ocr_cpp/wasm/>（Pages はルート配信で、
`/` には README が Jekyll で出る）。姉妹リポジトリと同じ形。

## 決めたことと、その理由

### データを自分で作る

外のデータセット（im2latex-100k など）を使わない。乱数で式木 → 組版して画像 →
**同じ木から** LaTeX。画像と答えがずれようがない。学習が進まないときに
「データが悪いのか実装が悪いのか」で悩まないで済むのが一番大きい。

### 同じ式を二度出さない

`gen` は出した式を覚えていて、重複を捨てる。やる前は 20000 件で中身 14449 種類、
別 seed の評価データの 32.9% が学習データと同じ式だった。`--exclude data/train` で
評価用を学習用から外せる。**この repo の精度の数字は、この状態で測ったもの**。

### フォントを揃える

`build/get_fonts.sh` が matplotlib 同梱の DejaVu を `fonts/` に置く。Windows の times.ttf と
Linux の DejaVu では字形が違い、揃えないと片方で作ったデータで学習したモデルが
もう片方の画像を読めない。揃えると Windows と Kaggle で `gen` の出力が md5 まで一致する
（確認済み: `data/train/labels.txt` も `images/000000.png` も同じ）。

### 重みは `.pt` 1 つ

`pure/ptio.hpp`（純 C++ の PyTorch .pt 読み書き）を使う。行列は `[in, out]` で持ち
`x @ W + b` で使う。Python 側は `nn.Linear` ではなく素の `nn.Parameter` を持つので、
転置の向きを両側で覚えなくて済む。

## 踏んだ罠（同じ所で止まらないように）

* **LayerNorm の eps**: engine 既定 1e-6 / torch 既定 1e-5。埋め込みの分散が 4e-4 しか
  ないので `1/sqrt(v+eps)` が 5e-3 動く。揃えるまでロジットが 3.6e-3 ずれていた。
  **生成される文字列は一致していた**ので、出力を見るだけでは気づけない。
  → `mdl::LN_EPS` に固定。借りてきた `tf_ops.hpp` は触らない（元 repo と分岐させない）。
* **float32 での勾配検算**: 1 個ずつ動かす差分は雑音に埋もれる。全部の重みを乱数方向に
  動かして `<g,v>` と中心差分を比べる（`selftest` がこれをやる）。
* **`ex::parse` は LaTeX を読まない**。読むのは `frac(1,2)` の記法。デモページの入力欄も
  この記法（組版と正解 LaTeX を同じ木から作るため）。
* **本家 pix2tex のデコーダは素の transformer ではない**: `attn_on_attn`（注意の出口が
  Linear → `nn.GLU`、sigmoid）と `ff_glu`（FF のゲートは gelu）。普通に書くと
  重みは全部読めて形も合うのに、出てくる LaTeX だけ壊れる。
* **本家は image_resizer 込みでないと読めない**: 無いと `E = mc^2` が
  `\underline{{{\cal I}}}` になる。幅当ての 21 クラス分類器（preact ResNetV2）も移植した。
  preact は norm が conv の前・stem に norm 無し・downsample も conv だけで、
  エンコーダ側の（preact でない）ResNetV2 とは**別物**。
* **heredoc がバックスラッシュを食う**（この機械）: `python - <<'PY'` に C++ の
  `'\\'` を入れると 1 段消える。パッチは `scratch/patch_*.py` にファイルとして書く。

## 次にやるなら

* 語彙と生成器を広げる（Σ・∫・添字・行列）。**組版と `ex::to_latex` を同時に直すこと**。
  いまは `\sum` が `sum(...)` と描かれてしまうので、生成器から意図的に外してある。
* 自前モデルの学習を C++ の CUDA ビルド（`build/nvcc.sh`）でも回す。いまの GPU 学習は
  Python 側でやっている（C++ CPU は 350 ms/サンプルで、実用にならない）。
* 本家モデルの微調整（いまは前向きだけ）。やるなら `pure/pix2tex.hpp` の前向きに
  逆伝播を通す必要がある（`group_norm` と `std_conv` に backward が無い）。
* WASM デモに本家モデルは載せていない（102 MB）。載せるなら量子化から。

## GPU（kbridge）の使い方

`kaggle_server_cpp` を使う。**URL は認証情報なので `scratch/kaggle_base.txt` にだけ置く**
（`scratch/` は gitignore 済み）。

```sh
cd ../kaggle_server_cpp/python && python -m kbridge.server --port 8787 &
curl -s -X POST localhost:8787/session -H 'Content-Type: application/json' \
     -d "{\"url\":\"$(cat scratch/kaggle_base.txt)\"}"
curl -s localhost:8787/gpu
curl -s -X POST localhost:8787/job -H 'Content-Type: application/json' -d '{"name":"train",
  "cmd":"cd /kaggle/working/latex_ocr_cpp && python tools/train.py train ..."}'
curl -s "localhost:8787/job/<id>/log?offset=0"
```

長い学習は必ず `/job`（切り離して起動、ログはファイル）。実際、**学習中に接続が何度か
切れた**が、ジョブは Kaggle 側で走り続けた。切れたら `/session` に URL を投げ直す。
