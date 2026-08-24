# latex_ocr_cpp — 数式の画像から LaTeX を読む（C++ と Python の両方で、学習も推論も）

数式の画像を入れると LaTeX が出る。**同じことを C++ でも Python でもできる**ようにしてあり、
どちらで学習した重みも、もう片方がそのまま読む。C++ 側は標準ライブラリだけ（外の
ライブラリを使わない自前の autograd）。ブラウザ用の [デモページ](https://yomei-o.github.io/latex_ocr_cpp/wasm/)
も同じコードを WebAssembly にしたもの。

入っているものは 2 つある。**別のものなので混ぜないこと**:

| | 自前のモデル | 本家の学習済みモデル |
|---|---|---|
| 何 | この repo で 1 から作って学習する | [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) の重みを借りて動かす |
| 大きさ | 55 万パラメータ（2.2 MB） | 2550 万パラメータ（102 MB） |
| 学習 | **できる**（C++ でも Python でも） | しない（前向きだけ） |
| 読めるもの | この repo が作る合成式 | 実物の論文の式 |
| 使う所 | WASM デモ、学習の実験 | CLI で実物を読ませる |

---

## すぐ試す

```sh
sh build/cc.sh pure/latexocr.cpp -o latexocr.exe     # Windows / MSVC
sh build/gcc.sh pure/latexocr.cpp -o latexocr.exe    # Linux / g++
sh build/get_fonts.sh                                # 描画に使うフォント（matplotlib 同梱の DejaVu）

./latexocr.exe gen --out data/train --n 30000 --seed 1
./latexocr.exe gen --out data/val --n 2000 --seed 777 --exclude data/train
./latexocr.exe train --data data/train --steps 2000 --export models/m.pt
./latexocr.exe infer --img data/val/images/000000.png --model models/m.pt
```

Python 側も同じことをする（重みは同じ `.pt`）:

```sh
python tools/train.py train --data data/train --val data/val --eval-every 2000 \
       --steps 30000 --batch 8 --device cuda --export models/best.pt
python tools/train.py infer --img data/val/images/000000.png --model models/best.pt
python tools/train.py eval  --data data/val --model models/best.pt
```

### 本家の学習済みモデルで実物を読む

```sh
sh build/get_pix2tex.sh                        # 重み 102 MB + 幅当て 19 MB + 語彙 24 KB
./latexocr.exe pix2tex --img formula.png       # C++
python tools/pix2tex.py --img formula.png      # Python（同じ文字列が出る）
```

---

## 測った値

### 自前のモデル（合成データ、Kaggle の T4 で 30000 step ≈ 57 分）

| | |
|---|---|
| 学習データ | 30000 件（**全部違う式**） |
| 評価データ | 2000 件（train と 1 つも被らない） |
| 見たことのない式での完全一致 | **1975 / 2000（98.8%）** |
| 学習の損失 | 4.46 → 0.000（30000 step、3388 秒） |
| WASM とネイティブの読み | 40 枚とも一致（`node wasm/test_node.js`） |

「完全一致」はトークン列が 1 つも違わないこと（部分点なし）。外した 25 件は、桁の多い数（`1407987` を `14079787`）と、入れ子の分数・指数（`p^{\frac{3}{2}}` を `p^{3}`）に偏っている。

### 本家の学習済みモデル

`scratch/` に本家（timm + x_transformers）を入れて突き合わせた結果:

| 比べたもの | 差 |
|---|---|
| エンコーダの出力 | 最大 4.5e-07 |
| ロジット | 最大 1.5e-05 |
| 貪欲な生成の id 列 | 完全一致 |
| 前処理した画像 | **ビット一致** |
| 試験画像 6 枚の LaTeX | C++ = Python = 本家、1 文字も違わない |

読めた例（matplotlib で組んだ画像）:

```
入力: \int_{0}^{\infty} e^{-x^{2}} dx = \frac{\sqrt{\pi}}{2}
読み: \int_{0}^{\infty}\mathrm{e}^{-\chi^{2}}d\chi={\frac{\sqrt{\pi}}{2}}
```

### 両言語の一致（自前のモデル）

| 比べたもの | 差 |
|---|---|
| ロジット | 2.95e-07 |
| 損失 | 0.00e+00 |
| 勾配（重み 1 本ずつ） | 最悪 2.98e-06 |
| 貪欲な生成 | 完全一致 |
| 重みの受け渡し（C++ ⇄ Python） | 両方向とも一致 |

---

## 作りの話

### データは自分で作る

外からデータセットを落としてこない。理由は 2 つ:

* **画像と LaTeX が必ず一致する。** 乱数で式木を作り、その木を組版して画像にし、
  **同じ木から** `ex::to_latex` で答えを書く。人手の注釈と違って、ずれようがない。
* いくらでも作れるので「データが足りない」と「モデルが弱い」を混同しないで済む。

同じ式を二度出さない（`gen` が覚えている）。これをやる前は 20000 件作っても中身は
14449 種類しかなく、別の seed で作った評価用データの **32.9% が学習データと同じ式**だった。
「見たことのある式を答えているだけ」で精度が上がってしまう。`--exclude data/train` で
評価用は学習用と 1 つも被らないようにできる。

フォントは `build/get_fonts.sh` で matplotlib 同梱の DejaVu を持ってくる。Windows の
`times.ttf` と Linux の DejaVu では字形が違うので、揃えないと**ローカルで作ったデータで
学習したモデルが、別の機械で作った画像を読めない**。揃えた結果、Windows と Kaggle で
`gen` が出す画像は md5 まで同じになる。

### 両言語をどう縛るか

「両方で動きます」と言うだけなら簡単だが、片方を直したときにもう片方がずれる。
なので**数で縛る**:

```sh
./latexocr.exe selftest                          # トークナイザの往復 + 勾配の検算
./latexocr.exe dump-parity --out scratch/parity.bin
python tools/parity/model.py --fixture scratch/parity.bin   # ロジット・損失・勾配・生成
python tools/parity/interchange.py                          # .pt の受け渡し（両方向）
python tools/parity/pix2tex.py                              # 本家モデルでの読み
```

`dump-parity` は C++ に「重みを初期化 → 画像を作る → 前向き → 損失 → 逆向き」をやらせて、
使った数を全部ファイルに書かせる。Python は**同じファイルを読んで**同じ計算をする。
入力が同じなので、違いが出たら実装の違いである（乱数の引き方でも画像の作り方でもない）。

**重み 1 本ずつの勾配まで比べる**のが要点。前向きだけ合わせても、学習は同じ道を行かない。

### 踏んだ罠

* **LayerNorm の eps。** この repo のエンジンは既定 1e-6、torch は 1e-5。埋め込みの分散が
  0.02² = 4e-4 しかないので、eps の差が `1/sqrt(v+eps)` を 5e-3 動かす。揃えるまで
  ロジットが 3.6e-3 ずれていた。**生成される文字列は一致していた**ので、出力を眺めるだけ
  では気づけない。
* **float32 での勾配検算。** 損失 4.7 に対して有効数字は 3e-7 しかなく、重み 1 個ずつを
  動かす差分は雑音に埋もれる（相対誤差 1.0 で FAIL する）。全部の重みを乱数方向に
  まとめて動かして `<g,v>` と中心差分を比べると測れる（実測 3.7e-03 で PASS）。
* **本家のデコーダは素の transformer ではない。** `attn_on_attn`（注意の出口が
  `Linear(512,512)` → `nn.GLU`、sigmoid ゲート）と `ff_glu`（FF のゲートは gelu）。
  ここを普通に書くと**重みは全部読めて形も合うのに、出てくる LaTeX だけが壊れる**。
* **本家は image_resizer が要る。** 同じ式でも字の大きさが学習時と違うと読めない:

  | | 読み |
  |---|---|
  | resizer 無し | `\underline{{{\cal I}}}\underline{{{\cal I}}}` |
  | resizer 有り | `E=m c^{2}` |

  なので幅を当てる 21 クラスの分類器（preact ResNetV2）も移植した。

---

## 中身

```
pure/            C++（標準ライブラリだけ）
  autograd.hpp     逆伝播つきテンソル（backend.hpp が CPU/CUDA の切り替え口）
  tok.hpp          自前モデルの語彙（82 トークン）
  expr.hpp         数式の木・有理数・構文解析・LaTeX 出力
  typeset*.hpp     式木 -> 画像（組版）
  gen_expr.hpp     乱数で式を作る
  data.hpp         画像と答えの組を作る（帯に収めるところも）
  model.hpp        自前モデル（CNN + transformer デコーダ）と .pt の読み書き
  pix2tex.hpp      本家の学習済みモデル（hybrid ViT + x_transformers デコーダ）
  bpe.hpp          本家の byte-level BPE（json も自前で読む）
  imgproc.hpp      本家と同じ前処理（PIL 互換の Lanczos まで込み）
  latexocr.cpp     CLI

tools/           Python（PyTorch）
  model.py         pure/model.hpp の鏡
  train.py         学習・推論・評価
  pix2tex.py       pure/pix2tex.hpp の鏡
  bpe.py           pure/bpe.hpp の鏡
  parity/          両言語を縛るテスト

wasm/            ブラウザ用（ここに出て、そのまま GitHub Pages で公開される）
build/           ビルドと取得のスクリプト
```

## ライセンス

このリポジトリのコードは MIT。本家 pix2tex の重みは
[lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)（MIT）のもので、
この repo には含めず `build/get_pix2tex.sh` で取ってくる。フォントは DejaVu
（Bitstream Vera 由来の許諾）で、同じく取ってくる。
