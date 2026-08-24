"""重みの行き来 — 片方で学習した .pt を、もう片方が読んで同じ答えを出すか。

tools/parity/model.py は「同じ重みなら同じ数」を縛る。こちらは**ファイルを跨ぐ**ほうを縛る:

  1. C++ で数 step 学習して .pt を書く   -> python が読んで推論
  2. python で数 step 学習して .pt を書く -> C++ が読んで推論

どちらも、両言語の貪欲生成が 1 文字も違わないことを確かめる。ここが通ると
「学習は速いほうでやり、推論は組み込みやすいほうでやる」ができる。

  python tools/parity/interchange.py --data data/train --val data/val --steps 20 --n 8
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit("失敗: %s" % " ".join(cmd))
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./latexocr.exe")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--val", default="data/val")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tmp", default="scratch")
    a = ap.parse_args()
    py = sys.executable

    imgs = sorted(os.listdir(os.path.join(ROOT, a.val, "images")))[:a.n]
    if not imgs:
        raise SystemExit("%s/images に画像がありません" % a.val)
    paths = ["%s/images/%s" % (a.val, i) for i in imgs]

    cpp_pt = "%s/ic_cpp.pt" % a.tmp
    py_pt = "%s/ic_py.pt" % a.tmp
    print("C++ で %d step 学習 -> %s" % (a.steps, cpp_pt))
    run([a.exe, "train", "--data", a.data, "--steps", str(a.steps), "--batch", "4",
         "--log-every", "0", "--export", cpp_pt])
    print("python で %d step 学習 -> %s" % (a.steps, py_pt))
    run([py, "tools/train.py", "train", "--data", a.data, "--steps", str(a.steps),
         "--batch", "4", "--log-every", "0", "--export", py_pt])

    ok = True
    for tag, pt in (("C++ が書いた .pt", cpp_pt), ("python が書いた .pt", py_pt)):
        bad = 0
        for p in paths:
            c = run([a.exe, "infer", "--model", pt, "--img", p])
            q = run([py, "tools/train.py", "infer", "--model", pt, "--img", p])
            if c != q:
                bad += 1
                if bad <= 3:
                    print("  NG %s\n     C++ %s\n     py  %s" % (os.path.basename(p), c, q))
        print("  %-18s %d/%d 一致  %s" % (tag, len(paths) - bad, len(paths),
                                          "ok" if bad == 0 else "NG"))
        ok = ok and bad == 0
    print("INTERCHANGE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
