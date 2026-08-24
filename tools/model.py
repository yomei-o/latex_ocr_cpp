"""LaTeX-OCR のモデル（Python 側）— pure/model.hpp の鏡。PyTorch で書く。

**同じ名前・同じ形の重み**を持つので、`.pt` をどちらの言語からでも読める。
行列は [in, out] で持ち、`x @ W + b` で使う（torch の nn.Linear は [out, in] なので、
素の nn.Parameter を持つ。転置の向きを両側で覚えなくて済む）。

食い違いは tools/parity/model.py が縛る（同じ重み・同じ入力で、ロジットと損失と勾配）。
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok  # noqa: E402


class Cfg:
    def __init__(self, H=64, W=256, d=128, heads=4, layers=2, ff=256, max_len=48, vocab=0):
        self.H, self.W, self.d, self.heads = H, W, d, heads
        self.layers, self.ff, self.max_len = layers, ff, max_len
        self.vocab = vocab

    def V(self):
        return self.vocab if self.vocab > 0 else tok.size()

    def enc_tokens(self):
        return (self.H // 8) * (self.W // 8)


class Model(nn.Module):
    """pure/model.hpp と同じ構成。重みの名前も同じ（`enc.c1.w` など）。

    PyTorch の Parameter 名に `.` は使えないので、内部では `__` に置き換えて持ち、
    state_dict を出し入れするときだけ戻す。
    """

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        V, d = c.V(), c.d
        self._names = []

        def add(name, *shape):
            p = nn.Parameter(torch.zeros(*shape))
            self.register_parameter(name.replace(".", "__"), p)
            self._names.append(name)
            return p

        add("enc.c1.w", 32, 1, 3, 3); add("enc.c1.b", 32)
        add("enc.c2.w", 64, 32, 3, 3); add("enc.c2.b", 64)
        add("enc.c3.w", d, 64, 3, 3); add("enc.c3.b", d)
        add("enc.pos", c.enc_tokens(), d)
        add("dec.emb", V, d)
        add("dec.pos", c.max_len, d)
        for l in range(c.layers):
            q = "dec.%d." % l
            add(q + "ln1.w", d); add(q + "ln1.b", d)
            add(q + "sa.qkv.w", d, 3 * d); add(q + "sa.qkv.b", 3 * d)
            add(q + "sa.proj.w", d, d); add(q + "sa.proj.b", d)
            add(q + "ln2.w", d); add(q + "ln2.b", d)
            add(q + "ca.q.w", d, d); add(q + "ca.q.b", d)
            add(q + "ca.kv.w", d, 2 * d); add(q + "ca.kv.b", 2 * d)
            add(q + "ca.proj.w", d, d); add(q + "ca.proj.b", d)
            add(q + "ln3.w", d); add(q + "ln3.b", d)
            add(q + "ff1.w", d, c.ff); add(q + "ff1.b", c.ff)
            add(q + "ff2.w", c.ff, d); add(q + "ff2.b", d)
        add("dec.lnf.w", d); add("dec.lnf.b", d)
        add("head.w", d, V); add("head.b", V)

    def p(self, name):
        return getattr(self, name.replace(".", "__"))

    def names(self):
        return list(self._names)

    # ---------------------------------------------------------------- 前向き
    def encode(self, img):
        """img [1, 1, H, W]（0..1、1 = 黒）-> [S, d]"""
        x = F.max_pool2d(F.relu(F.conv2d(img, self.p("enc.c1.w"), self.p("enc.c1.b"), 1, 1)), 2, 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, self.p("enc.c2.w"), self.p("enc.c2.b"), 1, 1)), 2, 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, self.p("enc.c3.w"), self.p("enc.c3.b"), 1, 1)), 2, 2)
        d, h, w = x.shape[1], x.shape[2], x.shape[3]
        # C++ は reshape([d, h*w]) して転置する。**同じ順**にすること（ここを間違えると
        # 絵の並びが変わって、両言語で別のモデルになる）
        x = x.reshape(d, h * w).transpose(0, 1)
        return x + self.p("enc.pos")

    def _attn(self, q_in, kv_in, qw, qb, kvw, kvb, pw, pb, mask):
        d = q_in.shape[1]
        hd = d // self.c.heads
        q = q_in @ qw + qb
        kv = kv_in @ kvw + kvb
        scale = 1.0 / math.sqrt(hd)
        outs = []
        for h in range(self.c.heads):
            qh = q[:, h * hd:(h + 1) * hd]
            kh = kv[:, h * hd:(h + 1) * hd]
            vh = kv[:, d + h * hd:d + (h + 1) * hd]
            s = (qh @ kh.transpose(0, 1)) * scale
            if mask is not None:
                s = s + mask
            outs.append(torch.softmax(s, dim=-1) @ vh)
        return torch.cat(outs, dim=1) @ pw + pb

    def _self_attn(self, x, qkvw, qkvb, pw, pb, mask):
        d = x.shape[1]
        hd = d // self.c.heads
        qkv = x @ qkvw + qkvb
        scale = 1.0 / math.sqrt(hd)
        outs = []
        for h in range(self.c.heads):
            q = qkv[:, h * hd:(h + 1) * hd]
            k = qkv[:, d + h * hd:d + (h + 1) * hd]
            v = qkv[:, 2 * d + h * hd:2 * d + (h + 1) * hd]
            s = (q @ k.transpose(0, 1)) * scale
            if mask is not None:
                s = s + mask
            outs.append(torch.softmax(s, dim=-1) @ v)
        return torch.cat(outs, dim=1) @ pw + pb

    def decode(self, enc, ids):
        """ids は入力トークン（BOS から）。返すのは [T, V] のロジット。"""
        T = len(ids)
        d = self.c.d
        idx = torch.tensor(ids, dtype=torch.long, device=enc.device)
        x = self.p("dec.emb")[idx] + self.p("dec.pos")[:T]
        mask = torch.full((T, T), -1e9, device=enc.device)
        mask = torch.triu(mask, diagonal=1)            # 0 か -1e9（C++ の causal_mask と同じ）
        for l in range(self.c.layers):
            q = "dec.%d." % l
            a = self._self_attn(
                F.layer_norm(x, (d,), self.p(q + "ln1.w"), self.p(q + "ln1.b")),
                self.p(q + "sa.qkv.w"), self.p(q + "sa.qkv.b"),
                self.p(q + "sa.proj.w"), self.p(q + "sa.proj.b"), mask)
            x = x + a
            b = self._attn(
                F.layer_norm(x, (d,), self.p(q + "ln2.w"), self.p(q + "ln2.b")), enc,
                self.p(q + "ca.q.w"), self.p(q + "ca.q.b"),
                self.p(q + "ca.kv.w"), self.p(q + "ca.kv.b"),
                self.p(q + "ca.proj.w"), self.p(q + "ca.proj.b"), None)
            x = x + b
            h = F.layer_norm(x, (d,), self.p(q + "ln3.w"), self.p(q + "ln3.b"))
            h = h @ self.p(q + "ff1.w") + self.p(q + "ff1.b")
            h = F.gelu(h) @ self.p(q + "ff2.w") + self.p(q + "ff2.b")
            x = x + h
        x = F.layer_norm(x, (d,), self.p("dec.lnf.w"), self.p("dec.lnf.b"))
        return x @ self.p("head.w") + self.p("head.b")

    def forward(self, img, ids):
        return self.decode(self.encode(img), ids)

    # ---------------------------------------------------------------- 貪欲な生成
    @torch.no_grad()
    def greedy(self, img, max_new=0):
        limit = max_new if max_new > 0 else self.c.max_len - 1
        enc = self.encode(img)
        ids = [tok.BOS]
        for _ in range(limit):
            logits = self.decode(enc, ids)
            nxt = int(torch.argmax(logits[-1]).item())
            if nxt == tok.EOS:
                break
            ids.append(nxt)
            if len(ids) >= self.c.max_len:
                break
        return ids


def ce_loss(logits, targets, device=None):
    """交差エントロピー（PAD を外す）。C++ の sq::ce_loss と同じ量。"""
    t = torch.tensor(targets, dtype=torch.long, device=logits.device)
    keep = t != tok.PAD
    lp = F.log_softmax(logits, dim=-1)
    picked = lp.gather(1, t.unsqueeze(1)).squeeze(1)
    return -(picked[keep].mean())


def init_params(m: Model, seed=1234):
    """C++ の mdl::init_params と**同じ乱数・同じ順**で初期化する。

    これが同じでないと「初期値から学習した結果」を両言語で比べられない。乱数は
    この repo の splitmix64（pure/rng.hpp）をそのまま Python でも回す。
    """
    state = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

    def next64():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def unit():
        return (next64() >> 11) * (1.0 / 9007199254740992.0)

    def normal():
        u1 = max(1e-12, unit())
        u2 = unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)

    with torch.no_grad():
        for n in m.names():
            p = m.p(n)
            if ".ln" in n:
                p.fill_(1.0 if n.endswith(".w") else 0.0)
                continue
            if ".pos" in n or ".emb" in n:
                flat = p.view(-1)
                for i in range(flat.numel()):
                    flat[i] = normal() * 0.02
                continue
            if p.dim() == 1:
                p.zero_()
                continue
            fan_in = p.shape[1] * p.shape[2] * p.shape[3] if p.dim() == 4 else p.shape[0]
            bound = 1.0 / math.sqrt(max(1, fan_in))
            flat = p.view(-1)
            for i in range(flat.numel()):
                flat[i] = (unit() * 2.0 - 1.0) * bound


def save(m: Model, path):
    sd = {n: m.p(n).detach().cpu() for n in m.names()}
    torch.save(sd, path, _use_new_zipfile_serialization=True)


def load(m: Model, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        miss = []
        for n in m.names():
            if n not in sd:
                miss.append(n)
                continue
            m.p(n).copy_(sd[n].reshape(m.p(n).shape))
    if miss:
        raise SystemExit("重みが足りません: %s" % ", ".join(miss[:5]))
    return m
