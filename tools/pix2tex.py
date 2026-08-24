"""本家 LaTeX-OCR（pix2tex）の学習済みモデルを、この repo のやり方で動かす（Python 側）。

**なぜ書き直すのか。** 本家は timm と x_transformers と albumentations に乗っている。
それをそのまま使うと「pip で入れた他人のモデルを呼んだ」だけになり、C++ 側に持っていけない。
ここでは重み（weights.pth）だけ借りて、前向きの計算は素の torch で書く。同じものを
pure/pix2tex.hpp にも書き、両者が同じ数を出すことを縛る（tools/parity/pix2tex.py）。

**構造の出どころは checkpoint そのもの**（コードの記憶ではなく、260 本のテンソルの名前と形）:

  encoder.patch_embed.backbone   ResNetV2（重み標準化 conv + GroupNorm(32) + ReLU）で 1/16 に
                                 stem 7x7/2 -> maxpool 3x3/2 -> stage [2,3,7] ブロック、64->1024ch
  encoder.patch_embed.proj       1x1 conv で 1024 -> 256（ここが ViT の patch 埋め込み）
  encoder.cls_token/pos_embed    pos_embed は [1, 505, 256] = cls + 12x42 の格子（最大 192x672 / 16）
  encoder.blocks.0..3            ふつうの ViT ブロック（pre-LN、qkv に bias、MLP は GELU）
  decoder.net                    x_transformers の Decoder 4 段 = (自己注意, 交差注意, FF) x 4
                                 **素の transformer ではない**:
                                   attn_on_attn  出力が Linear(512, 512) -> nn.GLU（sigmoid ゲート）
                                   ff_glu        FF が Linear(256, 2048) -> x * gelu(gate) -> Linear
                                 ここを普通の transformer で埋めると、重みは全部読めるのに
                                 出てくる LaTeX が壊れる（形が合うので気づきにくい）

  python tools/pix2tex.py --img formula.png --weights models/pix2tex/weights.pth
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Cfg:
    dim = 256
    heads = 8
    enc_depth = 4
    dec_depth = 4
    patch = 16
    max_w = 672
    max_h = 192
    min_w = 32
    min_h = 32
    num_tokens = 8000
    max_seq = 512
    backbone_layers = (2, 3, 7)
    bos, eos, pad = 1, 2, 0
    # 前処理の正規化（本家 pix2tex の test_transform と同じ値）
    mean = 0.7931
    std = 0.1738


# ---------------------------------------------------------------- 部品

def pad_same(x, k, s, value=0.0):
    """TensorFlow 流の "same" パディング（timm の Conv2dSame / MaxPool2dSame）。

    左右で厚さが違うことがある。**均等に割ると 1 画素ずれる**ので、端数は右下に寄せる。
    """
    ih, iw = x.shape[-2:]
    ph = max((math.ceil(ih / s) - 1) * s + k - ih, 0)
    pw = max((math.ceil(iw / s) - 1) * s + k - iw, 0)
    if ph or pw:
        x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2], value=value)
    return x


def std_conv(x, w, stride=1, eps=1e-6):
    """重み標準化つき conv（timm の StdConv2dSame）。**重みを出力チャネルごとに正規化してから**畳む。

    これを忘れると、重みは同じでも出る数が全部変わる（BatchNorm と違って推論時も効く）。
    """
    o = w.shape[0]
    f = w.reshape(1, o, -1)
    m = f.mean(dim=2, keepdim=True)
    v = f.var(dim=2, unbiased=False, keepdim=True)
    w = ((f - m) / torch.sqrt(v + eps)).reshape_as(w)
    return F.conv2d(pad_same(x, w.shape[-1], stride), w, None, stride)


def gn(x, w, b, groups=32, eps=1e-5, act=True):
    x = F.group_norm(x, groups, w, b, eps)
    return F.relu(x) if act else x


def ln(x, w, b, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


class P:
    """state_dict を名前で引くだけの入れ物（重みは torch の名前のまま持つ）。"""

    def __init__(self, sd, dev):
        self.sd = {k: v.to(dev) for k, v in sd.items()}

    def __call__(self, name):
        return self.sd[name]

    def has(self, name):
        return name in self.sd


# ---------------------------------------------------------------- エンコーダ

def bottleneck(x, p, q, stride):
    """timm ResNetV2 の（preact でない）Bottleneck。norm3 の後は活性化しないで足す。"""
    short = x
    if p.has(q + "downsample.conv.weight"):
        short = std_conv(x, p(q + "downsample.conv.weight"), stride)
        short = gn(short, p(q + "downsample.norm.weight"), p(q + "downsample.norm.bias"), act=False)
    x = std_conv(x, p(q + "conv1.weight"), 1)
    x = gn(x, p(q + "norm1.weight"), p(q + "norm1.bias"))
    x = std_conv(x, p(q + "conv2.weight"), stride)
    x = gn(x, p(q + "norm2.weight"), p(q + "norm2.bias"))
    x = std_conv(x, p(q + "conv3.weight"), 1)
    x = gn(x, p(q + "norm3.weight"), p(q + "norm3.bias"), act=False)
    return F.relu(x + short)


def backbone(img, p, c):
    b = "encoder.patch_embed.backbone."
    x = std_conv(img, p(b + "stem.conv.weight"), 2)
    x = gn(x, p(b + "stem.norm.weight"), p(b + "stem.norm.bias"))
    x = F.max_pool2d(pad_same(x, 3, 2, value=float("-inf")), 3, 2)
    for s, nblk in enumerate(c.backbone_layers):
        for i in range(nblk):
            stride = 2 if (i == 0 and s > 0) else 1
            x = bottleneck(x, p, "%sstages.%d.blocks.%d." % (b, s, i), stride)
    return x


def encoder(img, p, c):
    """画像 [1,1,H,W] -> トークン列 [1+h*w, 256]（先頭は cls）"""
    x = backbone(img, p, c)
    x = F.conv2d(x, p("encoder.patch_embed.proj.weight"), p("encoder.patch_embed.proj.bias"))
    _, _, fh, fw = x.shape
    x = x.reshape(x.shape[1], fh * fw).transpose(0, 1)          # [h*w, 256]
    x = torch.cat([p("encoder.cls_token").reshape(1, -1), x], 0)
    # 位置埋め込みは 12x42 の格子から**行ごとに切り出す**（横 672/16=42 の左端から w 個）。
    # ここを 1 次元に並べ替えると、幅の違う画像で位置がずれる。
    row = torch.arange(fh, device=x.device) * (c.max_w // c.patch - fw)
    idx = row.repeat_interleave(fw) + torch.arange(fh * fw, device=x.device)
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), idx + 1])
    x = x + p("encoder.pos_embed")[0, idx]
    for i in range(c.enc_depth):
        q = "encoder.blocks.%d." % i
        h = ln(x, p(q + "norm1.weight"), p(q + "norm1.bias"))
        T = h.shape[0]
        hd = c.dim // c.heads
        qkv = (h @ p(q + "attn.qkv.weight").t() + p(q + "attn.qkv.bias")).reshape(T, 3, c.heads, hd)
        qq, kk, vv = qkv[:, 0].transpose(0, 1), qkv[:, 1].transpose(0, 1), qkv[:, 2].transpose(0, 1)
        a = torch.softmax((qq @ kk.transpose(-1, -2)) * (hd ** -0.5), dim=-1) @ vv
        a = a.transpose(0, 1).reshape(T, c.dim)
        x = x + (a @ p(q + "attn.proj.weight").t() + p(q + "attn.proj.bias"))
        h = ln(x, p(q + "norm2.weight"), p(q + "norm2.bias"))
        h = F.gelu(h @ p(q + "mlp.fc1.weight").t() + p(q + "mlp.fc1.bias"))
        x = x + (h @ p(q + "mlp.fc2.weight").t() + p(q + "mlp.fc2.bias"))
    return ln(x, p("encoder.norm.weight"), p("encoder.norm.bias"))


# ---------------------------------------------------------------- デコーダ

def x_attn(x, ctx, p, q, heads, causal):
    """x_transformers の Attention（attn_on_attn つき）。

    出力が Linear(inner, dim*2) -> nn.GLU、つまり a * sigmoid(b)。**素の Linear ではない**。
    q,k,v に bias は無い（checkpoint に to_q.bias が無いのが根拠）。
    """
    T = x.shape[0]
    inner = p(q + "to_q.weight").shape[0]
    hd = inner // heads
    qq = (x @ p(q + "to_q.weight").t()).reshape(T, heads, hd).transpose(0, 1)
    kk = (ctx @ p(q + "to_k.weight").t()).reshape(-1, heads, hd).transpose(0, 1)
    vv = (ctx @ p(q + "to_v.weight").t()).reshape(-1, heads, hd).transpose(0, 1)
    s = (qq @ kk.transpose(-1, -2)) * (hd ** -0.5)
    if causal:
        s = s + torch.triu(torch.full((T, s.shape[-1]), float("-inf"), device=x.device), 1)
    o = (torch.softmax(s, dim=-1) @ vv).transpose(0, 1).reshape(T, inner)
    o = o @ p(q + "to_out.0.weight").t() + p(q + "to_out.0.bias")
    a, g = o.chunk(2, dim=-1)
    return a * torch.sigmoid(g)


def x_ff(x, p, q):
    """x_transformers の FF（ff_glu つき）。ゲートは **gelu**（注意側の sigmoid とは違う）。"""
    h = x @ p(q + "net.0.proj.weight").t() + p(q + "net.0.proj.bias")
    a, g = h.chunk(2, dim=-1)
    h = a * F.gelu(g)
    return h @ p(q + "net.2.weight").t() + p(q + "net.2.bias")


def decoder(enc, ids, p, c):
    """トークン列 -> ロジット [T, num_tokens]"""
    idx = torch.tensor(ids, dtype=torch.long, device=enc.device)
    x = p("decoder.net.token_emb.weight")[idx]
    x = x + p("decoder.net.pos_emb.emb.weight")[:len(ids)]
    for l in range(c.dec_depth):
        for k, kind in enumerate(("a", "c", "f")):
            q = "decoder.net.attn_layers.layers.%d." % (l * 3 + k)
            h = ln(x, p(q + "0.weight"), p(q + "0.bias"), eps=1e-5)
            if kind == "a":
                x = x + x_attn(h, h, p, q + "1.", c.heads, True)
            elif kind == "c":
                x = x + x_attn(h, enc, p, q + "1.", c.heads, False)
            else:
                x = x + x_ff(h, p, q + "1.")
    x = ln(x, p("decoder.net.norm.weight"), p("decoder.net.norm.bias"), eps=1e-5)
    return x @ p("decoder.net.to_logits.weight").t() + p("decoder.net.to_logits.bias")


@torch.no_grad()
def greedy(img, p, c, max_new=256):
    enc = encoder(img, p, c)
    ids = [c.bos]
    for _ in range(max_new):
        nxt = int(torch.argmax(decoder(enc, ids, p, c)[-1]).item())
        if nxt == c.eos:
            break
        ids.append(nxt)
        if len(ids) >= c.max_seq:
            break
    return ids[1:]


# ---------------------------------------------------------------- 前処理
#
# 本家は 3 つの部品でできている。**繰り返しの途中で順番が変わる**ので、まとめずに分けて持つ:
#   pil_pad(im)        明暗を伸ばして、インクのある所を切り出して、32 の倍数に白で埋める
#   minmax(im)         大きすぎたら縮める（672x192）、小さすぎたら白で足す（32x32）
#   to_tensor(im)      (x/255 - 0.7931) / 0.1738
# 最初は minmax(pil_pad(im))、resizer の繰り返しでは pil_pad(minmax(拡大縮小した元画像))。

def pil_pad(im, divable=32):
    from PIL import Image
    import numpy as np
    a = np.asarray(im.convert("LA"))
    a = a[..., 0] if a[..., 1].var() == 0 else 255 - a[..., 1]   # 透過つきなら alpha に字がある
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) * 255.0 if hi > lo else a * 0 + 255.0
    if a.mean() > 128:
        ink = a < 128                                            # 白地に黒字
    else:
        ink = a > 128
        a = 255.0 - a                                            # 黒地に白字 -> 反転
    if ink.any():
        ys, xs = np.where(ink)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    H = (h + divable - 1) // divable * divable
    W = (w + divable - 1) // divable * divable
    out = np.full((H, W), 255.0, dtype=np.float32)
    out[:h, :w] = a
    return Image.fromarray(out.astype(np.uint8))


def minmax(im, c):
    from PIL import Image
    import numpy as np
    r = max(im.size[0] / c.max_w, im.size[1] / c.max_h)
    if r > 1:
        size = np.array(im.size) // r
        im = im.resize(tuple(size.astype(int)), Image.BILINEAR)
    padded = [max(d, m) for d, m in zip(im.size, (c.min_w, c.min_h))]
    if padded != list(im.size):
        p = Image.new("L", padded, 255)
        p.paste(im, im.getbbox())
        im = p
    return im


def to_tensor(im, c):
    import numpy as np
    x = (np.asarray(im.convert("L"), dtype=np.float32) / 255.0 - c.mean) / c.std
    return torch.from_numpy(x).reshape(1, 1, *x.shape)


# ---------------------------------------------------------------- 画像の大きさを決めるモデル
#
# **これが精度をほぼ決める。** 同じ式でも、字の大きさが学習時と違うと本家でも読めない
# （実測: resizer 無しだと "E = mc^2" が \underline{{{\cal I}}} になり、有りだと E=m c^{2}）。
# image_resizer.pth は「この画像を横 (k+1)*32 画素にすべき」を当てる 21 クラスの分類器で、
# 中身は preact の ResNetV2 [2,3,3]（重み標準化 conv + GroupNorm、norm が conv の前に来る）。

def preact_block(x, p, q, stride):
    pre = gn(x, p(q + "norm1.weight"), p(q + "norm1.bias"))
    short = std_conv(pre, p(q + "downsample.conv.weight"), stride) \
        if p.has(q + "downsample.conv.weight") else x
    h = std_conv(pre, p(q + "conv1.weight"), 1)
    h = std_conv(gn(h, p(q + "norm2.weight"), p(q + "norm2.bias")), p(q + "conv2.weight"), stride)
    h = std_conv(gn(h, p(q + "norm3.weight"), p(q + "norm3.bias")), p(q + "conv3.weight"), 1)
    return h + short


@torch.no_grad()
def resizer_width(im, rp, c):
    """画像 -> ふさわしい横幅（画素）。本家 cli.py と同じ (argmax+1)*32。"""
    x = to_tensor(im, c)
    x = std_conv(x, rp("stem.conv.weight"), 2)
    x = F.max_pool2d(pad_same(x, 3, 2, value=float("-inf")), 3, 2)
    for s, nblk in enumerate((2, 3, 3)):
        for i in range(nblk):
            x = preact_block(x, rp, "stages.%d.blocks.%d." % (s, i), 2 if (i == 0 and s > 0) else 1)
    x = gn(x, rp("norm.weight"), rp("norm.bias"))
    x = x.mean(dim=(2, 3), keepdim=True)
    x = F.conv2d(x, rp("head.fc.weight"), rp("head.fc.bias")).flatten()
    return (int(x.argmax().item()) + 1) * 32


def read(path, p, c, rp=None, max_new=256):
    """画像のパス -> id 列。resizer を渡すと、本家と同じ繰り返しで大きさを決める。"""
    from PIL import Image
    src = Image.open(path)
    im = minmax(pil_pad(src), c)
    if rp is not None:
        rgb = src.convert("RGB").copy()
        r, w, h = 1.0, rgb.size[0], rgb.size[1]
        for _ in range(10):
            h = int(h * r)
            im = pil_pad(minmax(rgb.resize((w, h), Image.BILINEAR if r > 1 else Image.LANCZOS), c))
            w = resizer_width(im, rp, c)
            if w == im.size[0]:
                break
            r = w / im.size[0]
    return im, greedy(to_tensor(im, c), p, c, max_new)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--weights", default="models/pix2tex/weights.pth")
    ap.add_argument("--resizer", default="models/pix2tex/image_resizer.pth")
    ap.add_argument("--tokenizer", default="models/pix2tex/tokenizer.json")
    ap.add_argument("--no-resize", dest="no_resize", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", dest="max_new", type=int, default=256)
    a = ap.parse_args()
    c = Cfg()
    dev = torch.device(a.device)
    p = P(torch.load(a.weights, map_location="cpu", weights_only=True), dev)
    rp = None
    if not a.no_resize and os.path.exists(a.resizer):
        rp = P(torch.load(a.resizer, map_location="cpu", weights_only=True), dev)
    tk = bpe.Tokenizer(a.tokenizer)
    im, ids = read(a.img, p, c, rp, a.max_new)
    print("入力 %dx%d" % im.size)
    print(tk.decode(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
