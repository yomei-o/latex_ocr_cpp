#!/bin/sh
# 本家 LaTeX-OCR（pix2tex）の学習済みモデルを models/pix2tex/ に置く。
#
#   sh build/get_pix2tex.sh
#   ./latexocr.exe pix2tex --img formula.png
#   python tools/pix2tex.py --img formula.png
#
# 取ってくるもの（どれも lukas-blecher/LaTeX-OCR の v0.0.1 リリース）:
#   weights.pth        102 MB  読む本体（25.5M パラメータ）
#   image_resizer.pth   19 MB  画像を何画素幅にすべきかを当てる分類器
#   tokenizer.json      24 KB  byte-level BPE（語彙 1175）
#
# **image_resizer は要る。** 無いと同じ式でも読めない（実測: "E = mc^2" が
# \underline{{{\cal I}}} になる）。字の大きさが学習時と違うと当たらない、ということ。
#
# 重みは MIT ライセンス（本家 repo と同じ）。この repo には**含めない**（大きいので）。
set -e
cd "$(dirname "$0")/.."
mkdir -p models/pix2tex
B=https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1
T=https://raw.githubusercontent.com/lukas-blecher/LaTeX-OCR/main/pix2tex/model/dataset/tokenizer.json
get() {
  if [ -f "$2" ]; then echo "$2 は既にある"; return; fi
  echo "取得 $2"
  curl -sL -o "$2" "$1"
}
get "$B/weights.pth" models/pix2tex/weights.pth
get "$B/image_resizer.pth" models/pix2tex/image_resizer.pth
get "$T" models/pix2tex/tokenizer.json
ls -l models/pix2tex/
