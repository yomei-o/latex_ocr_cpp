#!/bin/sh
# 学習と描画で使うフォントを fonts/ に置く。
#
# **同じ字形で描かないと、機械が読む相手が変わってしまう**。Windows の times.ttf と
# Linux の DejaVu では字が違うので、ローカルで作ったデータで学習したモデルは
# Kaggle で作った画像を読めない（逆も同じ）。どこにでも同じ**ファイル**があるものを選ぶ。
#
# matplotlib は DejaVu を同梱していて、pip で入れた matplotlib なら中身は同一。
# Windows / Kaggle / Colab のどれでも同じ画像が出る。
#
#   sh build/get_fonts.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p fonts
python - <<'PY'
import os, shutil, sys
try:
    import matplotlib
except ImportError:
    sys.exit("matplotlib が要ります: pip install matplotlib")
d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for src, dst in (("DejaVuSerif.ttf", "fonts/math.ttf"),
                 ("DejaVuSerif-Italic.ttf", "fonts/math-italic.ttf")):
    p = os.path.join(d, src)
    if not os.path.exists(p):
        sys.exit("%s が見つかりません" % p)
    shutil.copyfile(p, dst)
    print("%s -> %s (%d bytes)" % (src, dst, os.path.getsize(dst)))
PY
