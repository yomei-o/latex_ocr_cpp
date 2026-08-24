#!/bin/sh
# ブラウザ用のビルド。emsdk はこの機械では C:\prog\emsdk\emsdk（EMSDK= で変えられる）。
# 初回だけ `cd $EMSDK && ./emsdk install latest && ./emsdk activate latest`。
#
#   sh build/emcc.sh                       # models/best.pt を焼き込む
#   MODEL=models/other.pt sh build/emcc.sh
#
# フォントと重みは --preload-file で MEMFS に入れる。こうすると pure/ 側の
# 「パスから読む」API をそのまま使えて、wasm 用の読み込み口を別に作らなくて済む
# （**同じコードが動いていることを保てる**のが目的）。
set -e
cd "$(dirname "$0")/.."
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
EMCC="$EMSDK/upstream/emscripten/emcc.py"
[ -f "$EMCC" ] || { echo "emcc.py が $EMCC に無い — cd $EMSDK && ./emsdk install latest"; exit 1; }
export EM_CONFIG="$EMSDK/.emscripten"

MODEL="${MODEL:-models/best.pt}"
FONT="${FONT:-fonts/math.ttf}"
OUT="${OUT:-docs/latexocr.js}"
[ -f "$MODEL" ] || { echo "$MODEL が無い（sh build/get_model.sh か、学習して --export）"; exit 1; }
[ -f "$FONT" ] || { echo "$FONT が無い（sh build/get_fonts.sh）"; exit 1; }
mkdir -p docs

python "$EMCC" -std=c++20 -O3 -Ipure -Ipure/third_party \
  -s MODULARIZE=1 -s EXPORT_NAME=createLatexOCR -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","HEAPU8"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_lx_w","_lx_h","_lx_ready","_lx_init","_lx_why","_lx_render","_lx_sample","_lx_fit","_lx_read"]' \
  --preload-file "$FONT@fonts/math.ttf" --preload-file "$MODEL@models/model.pt" \
  $EXTRA wasm/latexocr_wasm.cpp -o "$OUT"
ls -l "${OUT%.js}".* "${OUT%.js}.data" 2>/dev/null || true
echo "built $OUT"
