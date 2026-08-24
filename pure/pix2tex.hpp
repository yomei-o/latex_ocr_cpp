// 本家 LaTeX-OCR（pix2tex）の学習済みモデルを、この repo のエンジンで動かす。
//
// **なぜ自分で書くのか。** 本家は timm と x_transformers に乗っている。それを呼ぶだけでは
// C++ に持っていけない。ここでは重み（weights.pth, 25.5M パラメータ）だけ借りて、
// 前向きの計算はこのエンジンで書く。同じものが tools/pix2tex.py にもあり、
// tools/parity/pix2tex.py が「同じ画像で同じロジット」を縛る。
//
// **構造の出どころは checkpoint そのもの**（260 本のテンソルの名前と形）:
//
//   encoder.patch_embed.backbone  ResNetV2 で 1/16 に。**重み標準化 conv + GroupNorm(32)**
//                                 stem 7x7/2 -> maxpool 3x3/2 -> stage [2,3,7]、64 -> 1024 ch
//   encoder.patch_embed.proj      1x1 conv で 1024 -> 256（ViT の patch 埋め込みの代わり）
//   encoder.cls_token / pos_embed pos_embed は [505, 256] = cls + 12x42 の格子
//   encoder.blocks.0..3           ふつうの ViT ブロック（pre-LN、qkv に bias、MLP は GELU）
//   decoder.net                   x_transformers の Decoder 4 段 =(自己注意, 交差注意, FF)x4
//
// **素の transformer ではない**ところが 2 つある。ここを普通に書くと、重みは全部読めて
// 形も合うのに、出てくる LaTeX だけが壊れる（一番気づきにくい壊れ方）:
//
//   attn_on_attn  注意の出口が Linear(512, 512) -> nn.GLU、つまり a * sigmoid(b)
//   ff_glu        FF が Linear(256, 2048) -> a * gelu(b) -> Linear(1024, 256)
//                 ゲートが片方 sigmoid、片方 gelu。**同じ GLU ではない**
//
// 学習には使わない（前向きだけ。この repo で学習するのは pure/model.hpp のほう）。
// 借りた重みを直す道はここには無い、ということ。
#pragma once
#include "autograd.hpp"
#include "imgproc.hpp"
#include "linalg.hpp"
#include "nn_ops.hpp"
#include "ops2d.hpp"
#include "ptio.hpp"
#include "tf_ops.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace px {

struct Cfg {
  int dim = 256, heads = 8, enc_depth = 4, dec_depth = 4;
  int patch = 16, max_w = 672, max_h = 192, min_w = 32, min_h = 32;
  int max_seq = 512;
  int bos = 1, eos = 2, pad = 0;
  float mean = 0.7931f, sd = 0.1738f;     // 本家 test_transform の正規化
  int backbone[3] = {2, 3, 7};
};

// ---------------------------------------------------------------- 重み

struct Weights {
  std::map<std::string, Tensor> t;

  bool load(const std::string& path, std::string* why = nullptr) {
    const std::vector<pt::Tensor> ts = pt::load_pt(path);
    if (ts.empty()) { if (why) *why = path + " が読めません（.pt / .pth）"; return false; }
    for (const pt::Tensor& x : ts) t[x.name] = from_data(x.shape, x.data);
    return true;
  }
  bool has(const std::string& n) const { return t.find(n) != t.end(); }
  const Tensor& operator()(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) {
      // 綴りの出どころは checkpoint。落とすほうが、静かに 0 を使うより安い
      fprintf(stderr, "重み %s がありません\n", n.c_str());
      std::abort();
    }
    return it->second;
  }
  size_t count() const {
    size_t n = 0;
    for (const auto& kv : t) n += (size_t)kv.second->numel();
    return n;
  }
};

// ---------------------------------------------------------------- 部品

// TensorFlow 流の "same" パディング幅。**左右で厚さが違う**（端数は右下に寄る）
inline void same_pad(int64_t in, int64_t k, int64_t s, int64_t* lo, int64_t* hi) {
  const int64_t out = (in + s - 1) / s;
  const int64_t need = std::max<int64_t>((out - 1) * s + k - in, 0);
  *lo = need / 2;
  *hi = need - need / 2;
}

inline Tensor pad_lrtb(const Tensor& x, int64_t l, int64_t r, int64_t tp, int64_t bt, float v) {
  const int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  const int64_t OH = H + tp + bt, OW = W + l + r;
  Tensor o = make_tensor({N, C, OH, OW});
  std::fill(o->data.begin(), o->data.end(), v);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t h = 0; h < H; ++h)
        std::copy(&x->data[((n * C + c) * H + h) * W], &x->data[((n * C + c) * H + h) * W + W],
                  &o->data[((n * C + c) * OH + (h + tp)) * OW + l]);
  return o;
}

// 重み標準化つき conv（timm の StdConv2dSame）。**重みを出力チャネルごとに正規化してから**畳む。
// BatchNorm と違って推論時も効くので、忘れると出る数が全部変わる。
// 重みは変わらないので、正規化した結果は 1 度だけ作って使い回す。
inline Tensor std_weight(const Tensor& w, float eps = 1e-6f) {
  static std::map<const Node*, Tensor> cache;
  auto it = cache.find(w.get());
  if (it != cache.end()) return it->second;
  const int64_t O = w->shape[0], K = w->numel() / O;
  Tensor o = make_tensor(w->shape);
  for (int64_t c = 0; c < O; ++c) {
    const float* p = &w->data[(size_t)c * K];
    double m = 0;
    for (int64_t i = 0; i < K; ++i) m += p[i];
    m /= (double)K;
    double v = 0;
    for (int64_t i = 0; i < K; ++i) v += (p[i] - m) * (p[i] - m);
    v /= (double)K;                                        // 不偏でない分散（torch と同じ）
    const float inv = (float)(1.0 / std::sqrt(v + eps));
    for (int64_t i = 0; i < K; ++i) o->data[(size_t)c * K + i] = (float)((p[i] - m) * inv);
  }
  cache[w.get()] = o;
  return o;
}

inline Tensor std_conv(const Tensor& x, const Tensor& w, int64_t stride) {
  const int64_t k = w->shape[2];
  int64_t l, r, tp, bt;
  same_pad(x->shape[3], k, stride, &l, &r);
  same_pad(x->shape[2], k, stride, &tp, &bt);
  const Tensor px_ = (l || r || tp || bt) ? pad_lrtb(x, l, r, tp, bt, 0.f) : x;
  return conv2d(px_, std_weight(w), Tensor(), stride, 0, 1);
}

inline Tensor group_norm(const Tensor& x, const Tensor& g, const Tensor& b, int64_t groups = 32,
                         float eps = 1e-5f, bool act = true) {
  const int64_t N = x->shape[0], C = x->shape[1], HW = x->shape[2] * x->shape[3];
  const int64_t cg = C / groups;
  Tensor o = make_tensor(x->shape);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t gi = 0; gi < groups; ++gi) {
      const int64_t c0 = gi * cg;
      double m = 0;
      for (int64_t c = 0; c < cg; ++c)
        for (int64_t i = 0; i < HW; ++i) m += x->data[((n * C + c0 + c) * HW) + i];
      m /= (double)(cg * HW);
      double v = 0;
      for (int64_t c = 0; c < cg; ++c)
        for (int64_t i = 0; i < HW; ++i) {
          const double d = x->data[((n * C + c0 + c) * HW) + i] - m;
          v += d * d;
        }
      v /= (double)(cg * HW);
      const float inv = (float)(1.0 / std::sqrt(v + eps));
      for (int64_t c = 0; c < cg; ++c) {
        const float gg = g->data[c0 + c], bb = b->data[c0 + c];
        for (int64_t i = 0; i < HW; ++i) {
          float y = (float)((x->data[((n * C + c0 + c) * HW) + i] - m) * inv) * gg + bb;
          if (act && y < 0) y = 0;
          o->data[((n * C + c0 + c) * HW) + i] = y;
        }
      }
    }
  return o;
}

inline Tensor maxpool_same(const Tensor& x, int64_t k, int64_t s) {
  int64_t l, r, tp, bt;
  same_pad(x->shape[3], k, s, &l, &r);
  same_pad(x->shape[2], k, s, &tp, &bt);
  const Tensor px_ = pad_lrtb(x, l, r, tp, bt, -1e30f);    // 端は「無いもの」として扱う
  return maxpool2d(px_, k, s, 0);
}

// ---------------------------------------------------------------- エンコーダ

inline Tensor bottleneck(const Tensor& x, const Weights& w, const std::string& q, int64_t stride) {
  Tensor sc = x;
  if (w.has(q + "downsample.conv.weight")) {
    sc = std_conv(x, w(q + "downsample.conv.weight"), stride);
    sc = group_norm(sc, w(q + "downsample.norm.weight"), w(q + "downsample.norm.bias"), 32, 1e-5f,
                    false);
  }
  Tensor h = std_conv(x, w(q + "conv1.weight"), 1);
  h = group_norm(h, w(q + "norm1.weight"), w(q + "norm1.bias"));
  h = std_conv(h, w(q + "conv2.weight"), stride);
  h = group_norm(h, w(q + "norm2.weight"), w(q + "norm2.bias"));
  h = std_conv(h, w(q + "conv3.weight"), 1);
  h = group_norm(h, w(q + "norm3.weight"), w(q + "norm3.bias"), 32, 1e-5f, false);
  return relu(add(h, sc));
}

inline Tensor backbone(const Tensor& img, const Weights& w, const Cfg& c) {
  const std::string b = "encoder.patch_embed.backbone.";
  Tensor x = std_conv(img, w(b + "stem.conv.weight"), 2);
  x = group_norm(x, w(b + "stem.norm.weight"), w(b + "stem.norm.bias"));
  x = maxpool_same(x, 3, 2);
  for (int s = 0; s < 3; ++s)
    for (int i = 0; i < c.backbone[s]; ++i) {
      char q[96];
      snprintf(q, sizeof q, "%sstages.%d.blocks.%d.", b.c_str(), s, i);
      x = bottleneck(x, w, q, (i == 0 && s > 0) ? 2 : 1);
    }
  return x;
}

// 多頭注意 1 回分。x は [T, d]、ctx は [S, d]。qkv の重みは torch の [out, in] のまま。
inline Tensor heads_attn(const Tensor& q_all, const Tensor& k_all, const Tensor& v_all, int heads,
                         bool causal) {
  const int64_t T = q_all->shape[0], S = k_all->shape[0], inner = q_all->shape[1];
  const int64_t hd = inner / heads;
  const float scale = 1.f / std::sqrt((float)hd);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    const Tensor qh = slice_cols(q_all, h * hd, (h + 1) * hd);
    const Tensor kh = slice_cols(k_all, h * hd, (h + 1) * hd);
    const Tensor vh = slice_cols(v_all, h * hd, (h + 1) * hd);
    Tensor s = mul_scalar(matmul(qh, transpose2d(kh)), scale);
    if (causal) {
      for (int64_t i = 0; i < T; ++i)
        for (int64_t j = 0; j < S; ++j)
          if (j > i) s->data[i * S + j] = -1e30f;
    }
    outs.push_back(matmul(softmax_rows(s), vh));
  }
  return hcat(outs);
}

inline Tensor encode(const Tensor& img, const Weights& w, const Cfg& c) {
  Tensor f = backbone(img, w, c);
  f = conv2d(f, w("encoder.patch_embed.proj.weight"), w("encoder.patch_embed.proj.bias"), 1, 0, 1);
  const int64_t d = f->shape[1], fh = f->shape[2], fw = f->shape[3];
  Tensor x = transpose2d(reshape(f, {d, fh * fw}));        // [h*w, d]
  x = vcat({reshape(w("encoder.cls_token"), {1, d}), x});
  // 位置埋め込みは 12x42 の格子から**行ごとに**切り出す（横は左端から w 個）。
  // 1 次元に並べ替えると、幅の違う画像で位置がずれる。
  const int64_t gw = c.max_w / c.patch;
  Tensor pos = make_tensor({fh * fw + 1, d});
  const Tensor& pe = w("encoder.pos_embed");
  for (int64_t i = 0; i < fh * fw + 1; ++i) {
    int64_t src = 0;
    if (i > 0) {
      const int64_t k = i - 1;
      src = 1 + (k / fw) * gw + (k % fw);
    }
    std::copy(&pe->data[src * d], &pe->data[src * d + d], &pos->data[i * d]);
  }
  x = add(x, pos);
  for (int i = 0; i < c.enc_depth; ++i) {
    char q[64];
    snprintf(q, sizeof q, "encoder.blocks.%d.", i);
    const std::string p(q);
    Tensor h = layernorm(x, w(p + "norm1.weight"), w(p + "norm1.bias"), 1e-6f);
    const Tensor qkv = linear(h, w(p + "attn.qkv.weight"), w(p + "attn.qkv.bias"));
    const Tensor a = heads_attn(slice_cols(qkv, 0, c.dim), slice_cols(qkv, c.dim, 2 * c.dim),
                                slice_cols(qkv, 2 * c.dim, 3 * c.dim), c.heads, false);
    x = add(x, linear(a, w(p + "attn.proj.weight"), w(p + "attn.proj.bias")));
    h = layernorm(x, w(p + "norm2.weight"), w(p + "norm2.bias"), 1e-6f);
    h = gelu(linear(h, w(p + "mlp.fc1.weight"), w(p + "mlp.fc1.bias")));
    x = add(x, linear(h, w(p + "mlp.fc2.weight"), w(p + "mlp.fc2.bias")));
  }
  return layernorm(x, w("encoder.norm.weight"), w("encoder.norm.bias"), 1e-6f);
}

// ---------------------------------------------------------------- デコーダ

// x_transformers の Attention（attn_on_attn つき）。出口は Linear -> nn.GLU（a * sigmoid(b)）。
// q,k,v に bias は無い（checkpoint に to_q.bias が無いのが根拠）。
inline Tensor x_attn(const Tensor& x, const Tensor& ctx, const Weights& w, const std::string& q,
                     int heads, bool causal) {
  const Tensor qq = matmul(x, transpose2d(w(q + "to_q.weight")));
  const Tensor kk = matmul(ctx, transpose2d(w(q + "to_k.weight")));
  const Tensor vv = matmul(ctx, transpose2d(w(q + "to_v.weight")));
  const Tensor o = linear(heads_attn(qq, kk, vv, heads, causal), w(q + "to_out.0.weight"),
                          w(q + "to_out.0.bias"));
  const int64_t half = o->shape[1] / 2;
  return mul(slice_cols(o, 0, half), sigmoid(slice_cols(o, half, 2 * half)));
}

// x_transformers の FF（ff_glu つき）。ゲートは **gelu**（注意側の sigmoid とは違う）。
inline Tensor x_ff(const Tensor& x, const Weights& w, const std::string& q) {
  const Tensor h = linear(x, w(q + "net.0.proj.weight"), w(q + "net.0.proj.bias"));
  const int64_t half = h->shape[1] / 2;
  const Tensor g = mul(slice_cols(h, 0, half), gelu(slice_cols(h, half, 2 * half)));
  return linear(g, w(q + "net.2.weight"), w(q + "net.2.bias"));
}

inline Tensor decode(const Tensor& enc, const std::vector<int>& ids, const Weights& w,
                     const Cfg& c) {
  const int64_t T = (int64_t)ids.size();
  const Tensor& emb = w("decoder.net.token_emb.weight");
  const Tensor& pos = w("decoder.net.pos_emb.emb.weight");
  Tensor x = make_tensor({T, c.dim});
  for (int64_t i = 0; i < T; ++i)
    for (int64_t k = 0; k < c.dim; ++k)
      x->data[i * c.dim + k] = emb->data[(size_t)ids[i] * c.dim + k] + pos->data[i * c.dim + k];
  for (int l = 0; l < c.dec_depth; ++l)
    for (int k = 0; k < 3; ++k) {
      char q[80];
      snprintf(q, sizeof q, "decoder.net.attn_layers.layers.%d.", l * 3 + k);
      const std::string p(q);
      const Tensor h = layernorm(x, w(p + "0.weight"), w(p + "0.bias"), 1e-5f);
      if (k == 0) x = add(x, x_attn(h, h, w, p + "1.", c.heads, true));
      else if (k == 1) x = add(x, x_attn(h, enc, w, p + "1.", c.heads, false));
      else x = add(x, x_ff(h, w, p + "1."));
    }
  x = layernorm(x, w("decoder.net.norm.weight"), w("decoder.net.norm.bias"), 1e-5f);
  return linear(x, w("decoder.net.to_logits.weight"), w("decoder.net.to_logits.bias"));
}

// ---------------------------------------------------------------- 画像の大きさを決めるモデル
//
// **これが精度をほぼ決める。** image_resizer.pth は「この画像を横 (k+1)*32 画素にすべき」を
// 当てる 21 クラスの分類器で、中身は **preact** の ResNetV2 [2,3,3]。
// preact は norm が conv の**前**に来て、stem に norm が無く、downsample も conv だけ
// （上のエンコーダ側は preact でない ResNetV2。同じ ResNetV2 でも別物なので混ぜないこと）。

inline Tensor to_tensor(const ip::Img& im, const Cfg& c) {
  Tensor x = make_tensor({1, 1, im.h, im.w});
  for (size_t i = 0; i < im.p.size(); ++i)
    x->data[i] = ((float)im.p[i] / 255.f - c.mean) / c.sd;
  return x;
}

inline Tensor preact_block(const Tensor& x, const Weights& w, const std::string& q,
                           int64_t stride) {
  const Tensor pre = group_norm(x, w(q + "norm1.weight"), w(q + "norm1.bias"));
  const Tensor sc = w.has(q + "downsample.conv.weight")
                        ? std_conv(pre, w(q + "downsample.conv.weight"), stride)
                        : x;
  Tensor h = std_conv(pre, w(q + "conv1.weight"), 1);
  h = std_conv(group_norm(h, w(q + "norm2.weight"), w(q + "norm2.bias")), w(q + "conv2.weight"),
               stride);
  h = std_conv(group_norm(h, w(q + "norm3.weight"), w(q + "norm3.bias")), w(q + "conv3.weight"), 1);
  return add(h, sc);
}

inline int resizer_width(const ip::Img& im, const Weights& w, const Cfg& c) {
  Tensor x = to_tensor(im, c);
  x = std_conv(x, w("stem.conv.weight"), 2);
  x = maxpool_same(x, 3, 2);
  const int layers[3] = {2, 3, 3};
  for (int s = 0; s < 3; ++s)
    for (int i = 0; i < layers[s]; ++i) {
      char q[64];
      snprintf(q, sizeof q, "stages.%d.blocks.%d.", s, i);
      x = preact_block(x, w, q, (i == 0 && s > 0) ? 2 : 1);
    }
  x = group_norm(x, w("norm.weight"), w("norm.bias"));
  const int64_t C = x->shape[1], HW = x->shape[2] * x->shape[3];
  const Tensor& fw = w("head.fc.weight");                // [21, C, 1, 1]
  const Tensor& fb = w("head.fc.bias");
  int best = 0;
  float bestv = -1e30f;
  for (int64_t o = 0; o < fw->shape[0]; ++o) {
    double acc = fb->data[o];
    for (int64_t ch = 0; ch < C; ++ch) {
      double m = 0;
      for (int64_t i = 0; i < HW; ++i) m += x->data[ch * HW + i];
      acc += (m / (double)HW) * fw->data[o * C + ch];     // 平均プールしてから 1x1 conv
    }
    if ((float)acc > bestv) { bestv = (float)acc; best = (int)o; }
  }
  return (best + 1) * 32;
}

// 貪欲に読む。返すのは BOS を除いた id 列。
inline std::vector<int> greedy(const Tensor& img, const Weights& w, const Cfg& c, int max_new) {
  const Tensor enc = encode(img, w, c);
  std::vector<int> ids{c.bos};
  for (int i = 0; i < max_new && (int)ids.size() < c.max_seq; ++i) {
    const Tensor lg = decode(enc, ids, w, c);
    const int64_t V = lg->shape[1];
    const float* row = &lg->data[(lg->shape[0] - 1) * V];
    int best = 0;
    for (int64_t v = 1; v < V; ++v)
      if (row[v] > row[best]) best = (int)v;
    if (best == c.eos) break;
    ids.push_back(best);
  }
  return std::vector<int>(ids.begin() + 1, ids.end());
}

// 画像 -> id 列。resizer を渡すと、本家と同じ繰り返しで大きさを決める。
// 繰り返しは「今の幅で読ませる -> 望ましい幅を聞く -> 元画像をその幅に縮めて作り直す」。
// **元画像から作り直す**のが要点で、縮めた絵を更に縮めると字が潰れる。
inline std::vector<int> read(const ip::Img& src, const Weights& w, const Cfg& c,
                             const Weights* rz, int max_new, ip::Img* used = nullptr) {
  ip::Img im = ip::minmax(ip::pil_pad(src), c.max_w, c.max_h, c.min_w, c.min_h);
  if (rz) {
    double r = 1.0;
    int tw = src.w, th = src.h;
    for (int i = 0; i < 10; ++i) {
      th = (int)(th * r);
      const ip::Img scaled = ip::resize(src, tw, th, r > 1 ? ip::BILINEAR : ip::LANCZOS);
      im = ip::pil_pad(ip::minmax(scaled, c.max_w, c.max_h, c.min_w, c.min_h));
      const int want = resizer_width(im, *rz, c);
      if (want == im.w) break;
      r = (double)want / (double)im.w;
      tw = want;
    }
  }
  if (used) *used = im;
  return greedy(to_tensor(im, c), w, c, max_new);
}

}  // namespace px
