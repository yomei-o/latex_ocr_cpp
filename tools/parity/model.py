"""モデルのパリティ — 同じ重み・同じ画像を入れたら C++ と Python が同じ数を出すか。

突き合わせるもの:
  1. **ロジット**（[T, V] の全部）      前向きが同じか
  2. **損失**                          交差エントロピーの取り方が同じか
  3. **勾配**（重み 1 つ 1 つ）        逆向きが同じか。ここが合えば学習は同じ道を行く
  4. **貪欲な生成の結果**              推論が同じか

やり方: C++ に `--dump-parity` で「重みを初期化 -> 画像を作る -> 前向き -> 損失 -> 逆向き」を
やらせ、使った数を全部ファイルに書かせる。Python は**同じファイルを読んで**同じ計算をする。
入力が同じなので、違いが出たら実装の違いである（乱数の引き方でも画像の作り方でもない）。

  latexocr dump-parity --out scratch/parity.bin
  python tools/parity/model.py --fixture scratch/parity.bin
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import model as M      # noqa: E402
import tok             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_fixture(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"LTXPAR01":
        raise SystemExit("%s は LTXPAR01 の fixture ではありません" % path)
    off = 8
    H, W, d, heads, layers, ff, max_len, V, T = struct.unpack_from("<9i", blob, off)
    off += 36
    fx = {"cfg": M.Cfg(H=H, W=W, d=d, heads=heads, layers=layers, ff=ff, max_len=max_len,
                       vocab=V)}
    n_img = H * W
    fx["img"] = np.frombuffer(blob, np.float32, n_img, off).copy()
    off += 4 * n_img
    fx["in_ids"] = list(struct.unpack_from("<%di" % T, blob, off))
    off += 4 * T
    fx["tgt"] = list(struct.unpack_from("<%di" % T, blob, off))
    off += 4 * T
    (nw,) = struct.unpack_from("<i", blob, off)
    off += 4
    w = {}
    g = {}
    for _ in range(nw):
        (ln,) = struct.unpack_from("<i", blob, off)
        off += 4
        name = blob[off:off + ln].decode()
        off += ln
        (nd,) = struct.unpack_from("<i", blob, off)
        off += 4
        shape = list(struct.unpack_from("<%di" % nd, blob, off))
        off += 4 * nd
        n = int(np.prod(shape))
        w[name] = np.frombuffer(blob, np.float32, n, off).reshape(shape).copy()
        off += 4 * n
        g[name] = np.frombuffer(blob, np.float32, n, off).reshape(shape).copy()
        off += 4 * n
    fx["w"], fx["g"] = w, g
    n_log = T * V
    fx["logits"] = np.frombuffer(blob, np.float32, n_log, off).reshape(T, V).copy()
    off += 4 * n_log
    (fx["loss"],) = struct.unpack_from("<f", blob, off)
    off += 4
    (ng,) = struct.unpack_from("<i", blob, off)
    off += 4
    fx["greedy"] = list(struct.unpack_from("<%di" % ng, blob, off))
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tol", type=float, default=2e-4)
    a = ap.parse_args()
    fx = read_fixture(a.fixture)
    c = fx["cfg"]
    m = M.Model(c)
    with torch.no_grad():
        for n in m.names():
            if n not in fx["w"]:
                raise SystemExit("fixture に %s がありません" % n)
            m.p(n).copy_(torch.from_numpy(fx["w"][n]))

    img = torch.from_numpy(fx["img"]).reshape(1, 1, c.H, c.W)
    logits = m(img, fx["in_ids"])
    loss = M.ce_loss(logits, fx["tgt"])
    m.zero_grad()
    loss.backward()

    print("d %d heads %d layers %d V %d、入力 %d トークン、重み %d 本"
          % (c.d, c.heads, c.layers, c.V(), len(fx["in_ids"]), len(fx["w"])))
    ok = True

    lg = logits.detach().numpy()
    rel = float(np.abs(lg - fx["logits"]).max()) / max(1e-9, float(np.abs(fx["logits"]).max()))
    ok = ok and rel <= a.tol
    print("  ロジット  最大の差 / 最大値 = %.2e  %s" % (rel, "ok" if rel <= a.tol else "NG"))

    dl = abs(float(loss.item()) - fx["loss"]) / max(1e-9, abs(fx["loss"]))
    ok = ok and dl <= a.tol
    print("  損失      C++ %.6f  python %.6f  rel %.2e  %s"
          % (fx["loss"], float(loss.item()), dl, "ok" if dl <= a.tol else "NG"))

    worst, wname = 0.0, ""
    for n in m.names():
        gp = m.p(n).grad
        gp = np.zeros_like(fx["g"][n]) if gp is None else gp.detach().numpy()
        scale = max(1e-8, float(np.abs(fx["g"][n]).max()))
        r = float(np.abs(gp - fx["g"][n]).max()) / scale
        if r > worst:
            worst, wname = r, n
    ok = ok and worst <= a.tol * 50           # 勾配は掛け算が積み重なるので少し緩める
    print("  勾配      最悪 %.2e（%s）  %s" % (worst, wname, "ok" if worst <= a.tol * 50 else "NG"))

    got = m.greedy(img)
    same = got == fx["greedy"]
    ok = ok and same
    print("  貪欲生成  C++ %s" % tok.decode(fx["greedy"]))
    print("            py  %s  %s" % (tok.decode(got), "ok" if same else "NG"))

    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
