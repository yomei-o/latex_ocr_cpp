"""本家の学習済みモデル（pix2tex）のパリティ — C++ と python が同じ LaTeX を出すか。

比べるのは**最後の文字列**にしてある。前処理に拡大縮小が入るので、python 側（PIL）と
C++ 側（自前の PIL 互換）で画素が 1 階調ずれることがある。**そこを合わせるのが目的ではなく**、
読みが変わらないことが目的なので、判定は「同じ大きさに落ち着いたか」「同じ LaTeX か」で見る。

モデルそのものの一致は別に測ってある（scratch/ref_dump.py で本家の出力と突き合わせ:
エンコーダ 4.5e-07、ロジット 1.5e-05、貪欲な生成は完全一致）。

  python tools/parity/pix2tex.py --dir scratch/testimg
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scratch/testimg")
    ap.add_argument("--exe", default="./latexocr.exe")
    ap.add_argument("--weights", default="models/pix2tex/weights.pth")
    ap.add_argument("--resizer", default="models/pix2tex/image_resizer.pth")
    ap.add_argument("--tokenizer", default="models/pix2tex/tokenizer.json")
    a = ap.parse_args()

    import torch
    import bpe
    import pix2tex as PX

    d = os.path.join(ROOT, a.dir)
    imgs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not imgs:
        raise SystemExit("%s に png がありません" % a.dir)
    c = PX.Cfg()
    p = PX.P(torch.load(a.weights, map_location="cpu", weights_only=True), torch.device("cpu"))
    rp = PX.P(torch.load(a.resizer, map_location="cpu", weights_only=True), torch.device("cpu")) \
        if os.path.exists(a.resizer) else None
    tk = bpe.Tokenizer(a.tokenizer)

    ok = True
    for name in imgs:
        path = os.path.join(a.dir, name).replace("\\", "/")
        im, ids = PX.read(os.path.join(d, name), p, c, rp)
        py = bpe.post_process(tk.decode(ids))
        out = subprocess.run([a.exe, "pix2tex", "--img", path, "--weights", a.weights,
                              "--resizer", a.resizer, "--tokenizer", a.tokenizer],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        lines = [l for l in out.stdout.strip().split("\n") if l]
        cpp_size = lines[0].replace("入力 ", "") if lines else "?"
        cpp = lines[-1] if lines else ""
        same = cpp == py
        ok = ok and same
        print("  %-8s %-9s %s" % (name, cpp_size, "ok" if same else "ちがう"))
        if not same:
            print("     C++ %s\n     py  %s" % (cpp, py))
        else:
            print("     %s" % py)
    print("PIX2TEX PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
