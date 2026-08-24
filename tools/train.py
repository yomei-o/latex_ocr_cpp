"""LaTeX-OCR の学習と推論（Python 側）— pure/latexocr.cpp の train / infer / eval の鏡。

  python tools/train.py train --data data/train --steps 200 --export models/py.pt
  python tools/train.py infer --img x.png --model models/py.pt
  python tools/train.py eval  --data data/val --model models/py.pt --limit 100

**重みは C++ と同じ .pt**。片方で学習したものをもう片方で動かせる（tools/parity/model.py が
同じ数を出すことを縛っている）。GPU があれば --device cuda で速くなる。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import model as M      # noqa: E402
import tok             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cfg_of(a):
    return M.Cfg(H=a.h, W=a.w, d=a.dim, heads=a.heads, layers=a.layers, ff=a.ff,
                 max_len=a.max_len)


def read_labels(d):
    """labels.txt を読む（C++ の read_labels と同じ形式: ファイル名 <TAB> LaTeX）。"""
    out = []
    p = os.path.join(d, "labels.txt")
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "\t" not in line:
                continue
            name, latex = line.split("\t", 1)
            out.append((os.path.join(d, "images", name), latex))
    return out


def load_gray(path, H, W):
    """png を読んで [H*W] の 0..1（1 = 黒）にする（C++ の load_gray と同じ向き）。"""
    from PIL import Image
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.float32)
    out = np.zeros((H, W), dtype=np.float32)
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = (255.0 - a[:h, :w]) / 255.0
    return out.reshape(-1)


def load_u8(path, H, W):
    """同じものを uint8 で持つ（キャッシュ用）。float で 30000 枚持つと 1.9 GB になる。"""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    out = np.zeros((H, W), dtype=np.uint8)
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = 255 - a[:h, :w]
    return out.reshape(-1)


def lr_at(step, a):
    """warmup してから cosine で落とす。

    transformer は**最初の数百 step が一番壊れやすい**（注意の softmax が飽和したまま
    大きい step を踏むと戻ってこない）。頭を小さく入って、後半で細かく詰める。
    C++ 側（cmd_train）も同じ式を使う。
    """
    if step < a.warmup:
        return a.lr * (step + 1) / max(1, a.warmup)
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1.0 + math.cos(math.pi * min(1.0, t)))


def train(a):
    c = cfg_of(a)
    dev = torch.device(a.device)
    items = read_labels(a.data)
    if a.limit > 0:
        items = items[:a.limit]
    if not items:
        raise SystemExit("%s/labels.txt が読めません" % a.data)
    m = M.Model(c).to(dev)
    if a.init:
        M.load(m, a.init)
        m.to(dev)
    else:
        M.init_params(m, a.seed)
        m.to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    print("data %d 件、パラメータ %d、d %d heads %d layers %d、batch %d、%d step、lr %g、%s"
          % (len(items), nparam, c.d, c.heads, c.layers, a.batch, a.steps, a.lr, dev))
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(a.seed)
    cache = {}
    val = read_labels(a.val)[:a.val_n] if a.val else []
    first = last = 0.0
    best = -1.0
    t0 = time.time()

    def save_to(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sd = {n: m.p(n).detach().cpu() for n in m.names()}
        torch.save(sd, path, _use_new_zipfile_serialization=True)

    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=False)
        tot, used = 0.0, 0
        for _ in range(a.batch):
            path, latex = items[int(rng.integers(len(items)))]
            if path not in cache:
                cache[path] = load_u8(path, c.H, c.W)
            ids = tok.encode(latex, bos=True, eos=True)
            if len(ids) < 2 or len(ids) > c.max_len:
                continue
            img = torch.from_numpy(cache[path].astype(np.float32) / 255.0)
            logits = m(img.reshape(1, 1, c.H, c.W).to(dev), ids[:-1])
            loss = M.ce_loss(logits, ids[1:])
            loss.backward()
            tot += float(loss.item())
            used += 1
        if not used:
            print("step %d: 使える件が無い" % step)
            break
        for p in m.parameters():                        # batch で平均（C++ と同じ）
            if p.grad is not None:
                p.grad /= used
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip))
        opt.step()
        l = tot / used
        if step == 0:
            first = l
        last = l
        if a.log_every > 0 and (step % a.log_every == 0 or step == a.steps - 1):
            print("step %d: loss %.4f |g| %.2f lr %.2e (%.0f 秒)"
                  % (step, l, gn, lr_at(step, a), time.time() - t0))
            sys.stdout.flush()
        # **途中で val を読ませる**。損失が下がっても文字列が合うとは限らないので、
        # 学習中に見るのは「完全一致」のほうにする。良くなった時だけ書き出す。
        if val and a.eval_every > 0 and (step + 1) % a.eval_every == 0:
            m.eval()
            ok = 0
            with torch.no_grad():
                for p_, latex in val:
                    if p_ not in cache:
                        cache[p_] = load_u8(p_, c.H, c.W)
                    im = torch.from_numpy(cache[p_].astype(np.float32) / 255.0)
                    got = tok.decode(m.greedy(im.reshape(1, 1, c.H, c.W).to(dev)))
                    ok += got == tok.decode(tok.encode(latex))
            m.train()
            acc = ok / len(val)
            mark = ""
            if a.export and acc > best:
                best = acc
                save_to(a.export)
                mark = " -> %s" % a.export
            print("step %d: val 完全一致 %d/%d (%.1f%%)%s"
                  % (step, ok, len(val), 100 * acc, mark))
            sys.stdout.flush()
    print("loss %.4f -> %.4f（%d step、%.1f 秒）" % (first, last, a.steps, time.time() - t0))
    if a.export and best < 0:                            # val を見ていないなら最後の重みを書く
        save_to(a.export)
        print("wrote %s" % a.export)
    return 0


def infer(a):
    c = cfg_of(a)
    m = M.load(M.Model(c), a.model).eval()
    img = torch.from_numpy(load_gray(a.img, c.H, c.W)).reshape(1, 1, c.H, c.W)
    print(tok.decode(m.greedy(img)))
    return 0


def evaluate(a):
    c = cfg_of(a)
    dev = torch.device(a.device)
    m = M.load(M.Model(c), a.model).eval().to(dev)
    items = read_labels(a.data)
    if a.limit > 0:
        items = items[:a.limit]
    ok = 0
    n = 0
    shown = 0
    for path, latex in items:
        img = torch.from_numpy(load_gray(path, c.H, c.W)).reshape(1, 1, c.H, c.W).to(dev)
        n += 1
        got = tok.decode(m.greedy(img))
        want = tok.decode(tok.encode(latex))
        if got == want:
            ok += 1
        elif a.show_fail and shown < 10:
            shown += 1
            print("  NG  正解 %-28s 読み %s" % (want, got))
    print("完全一致: %d / %d（%.1f%%）" % (ok, n, 100.0 * ok / n if n else 0.0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "infer", "eval"])
    ap.add_argument("--data", default="")
    ap.add_argument("--img", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--export", default="")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", dest="lr_min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val", default="")
    ap.add_argument("--val-n", dest="val_n", type=int, default=200)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", dest="log_every", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show-fail", dest="show_fail", action="store_true")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--max-len", dest="max_len", type=int, default=48)
    a = ap.parse_args()
    if a.cmd == "train":
        return train(a)
    if a.cmd == "infer":
        return infer(a)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
