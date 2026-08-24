// LaTeX-OCR のモデル — 画像 -> LaTeX のトークン列。
//
// 形（小さく作る。学習も推論も CPU で回せることを優先する）:
//
//   encoder  画像 [1,1,H,W] を CNN で 1/8 に落として、格子を並べて列にする
//            conv(1->32) relu pool2 / conv(32->64) relu pool2 / conv(64->d) relu pool2
//            -> [H/8 * W/8, d] + 位置の埋め込み（学習する）
//   decoder  L 層。層ごとに「因果 self-attention -> encoder への cross-attention -> FFN」
//            前に LayerNorm を置く（pre-norm。学習が安定する）
//   head     最後の LayerNorm -> [d, V] で語彙のロジット
//
// **重みは名前つきで持ち、PyTorch の state_dict と同じ名前・同じ形にする**（pure/ptio.hpp が
// .pt を直接読み書きできるので、片方の言語で学習した重みをもう片方でそのまま動かせる）。
// 行列は [in, out] で持つ（torch の nn.Linear は [out, in] なので、Python 側は
// nn.Parameter を素で持って x @ W とする。転置の向きを両側で覚えなくて済む）。
#pragma once
#include "autograd.hpp"
#include "linalg.hpp"
#include "nn_ops.hpp"
#include "ops2d.hpp"
#include "ptio.hpp"
#include "rng.hpp"
#include "seq_ops.hpp"
#include "tf_ops.hpp"
#include "tok.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace mdl {

// **LayerNorm の eps は torch の既定に合わせる**（1e-5）。engine の既定は 1e-6 だが、
// 埋め込みの分散は 0.02^2 = 4e-4 しかないので、eps の差が 1/sqrt(v+eps) で 5e-3 動く
// （実測: これを揃えるまで両言語のロジットが 3.6e-3 ずれていた）。
inline constexpr float LN_EPS = 1e-5f;
inline Tensor ln(const Tensor& x, const Tensor& g, const Tensor& b) {
  return layernorm(x, g, b, LN_EPS);
}


struct Cfg {
  int H = 64, W = 256;        // 入力の帯
  int d = 128;                // 埋め込みの次元
  int heads = 4;
  int layers = 2;
  int ff = 256;
  int max_len = 48;           // 出せるトークンの上限
  int vocab = 0;              // 0 なら tok::size()
  int V() const { return vocab > 0 ? vocab : tok::size(); }
  int enc_tokens() const { return (H / 8) * (W / 8); }
};

// 名前つきの重みの入れ物。**作る順ではなく名前で引く**（.pt と行き来するため）
struct Params {
  std::map<std::string, Tensor> t;
  std::vector<std::string> order;                      // 作った順（表示と初期化のため）

  Tensor add(const std::string& name, std::vector<int64_t> shape) {
    Tensor x = make_tensor(shape, true);
    t[name] = x;
    order.push_back(name);
    return x;
  }
  Tensor get(const std::string& name) const {
    const std::map<std::string, Tensor>::const_iterator it = t.find(name);
    return it == t.end() ? Tensor() : it->second;
  }
  std::vector<Tensor> all() const {
    std::vector<Tensor> v;
    for (const std::string& n : order) v.push_back(t.at(n));
    return v;
  }
  size_t count() const {
    size_t n = 0;
    for (const std::string& s : order) n += (size_t)t.at(s)->numel();
    return n;
  }
};

// torch の既定に合わせた初期化。
//   * 行列（Linear 相当）は kaiming_uniform_(a=sqrt(5)) と同じ ±1/sqrt(in)
//   * conv も同じ規則（fan_in = Cin*kh*kw）
//   * 埋め込みは normal(0, 1) ではなく **正規 0.02**（学習が安定する。GPT 系の定石）
//   * LayerNorm は gamma=1, beta=0
inline void init_params(Params& p, Rng& rng) {
  const auto uni = [&](Tensor& x, double bound) {
    for (float& v : x->data) v = (float)((rng.unit() * 2.0 - 1.0) * bound);
  };
  // Box-Muller（この repo の splitmix64 から。Python 側も同じ式にする）
  const auto normal = [&]() {
    const double u1 = std::max(1e-12, rng.unit()), u2 = rng.unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  };
  for (const std::string& n : p.order) {
    Tensor x = p.t[n];
    if (n.find(".ln") != std::string::npos) {          // LayerNorm は gamma=1, beta=0
      const bool is_w = n.size() > 2 && n.substr(n.size() - 2) == ".w";
      for (float& v : x->data) v = is_w ? 1.f : 0.f;
      continue;
    }
    if (n.find(".pos") != std::string::npos || n.find(".emb") != std::string::npos) {
      for (float& v : x->data) v = (float)(normal() * 0.02);
      continue;
    }
    if (x->shape.size() == 1) {                        // bias
      for (float& v : x->data) v = 0.f;
      continue;
    }
    int64_t fan_in = x->shape.size() == 4
                         ? x->shape[1] * x->shape[2] * x->shape[3]   // conv: Cin*kh*kw
                         : x->shape[0];                              // matmul: [in, out]
    uni(x, 1.0 / std::sqrt((double)std::max<int64_t>(1, fan_in)));
  }
}

inline void build(Params& p, const Cfg& c) {
  const int V = c.V();
  p.add("enc.c1.w", {32, 1, 3, 3});
  p.add("enc.c1.b", {32});
  p.add("enc.c2.w", {64, 32, 3, 3});
  p.add("enc.c2.b", {64});
  p.add("enc.c3.w", {(int64_t)c.d, 64, 3, 3});
  p.add("enc.c3.b", {(int64_t)c.d});
  p.add("enc.pos", {(int64_t)c.enc_tokens(), (int64_t)c.d});
  p.add("dec.emb", {(int64_t)V, (int64_t)c.d});
  p.add("dec.pos", {(int64_t)c.max_len, (int64_t)c.d});
  for (int l = 0; l < c.layers; ++l) {
    const std::string q = "dec." + std::to_string(l) + ".";
    p.add(q + "ln1.w", {(int64_t)c.d});
    p.add(q + "ln1.b", {(int64_t)c.d});
    p.add(q + "sa.qkv.w", {(int64_t)c.d, (int64_t)(3 * c.d)});
    p.add(q + "sa.qkv.b", {(int64_t)(3 * c.d)});
    p.add(q + "sa.proj.w", {(int64_t)c.d, (int64_t)c.d});
    p.add(q + "sa.proj.b", {(int64_t)c.d});
    p.add(q + "ln2.w", {(int64_t)c.d});
    p.add(q + "ln2.b", {(int64_t)c.d});
    p.add(q + "ca.q.w", {(int64_t)c.d, (int64_t)c.d});
    p.add(q + "ca.q.b", {(int64_t)c.d});
    p.add(q + "ca.kv.w", {(int64_t)c.d, (int64_t)(2 * c.d)});
    p.add(q + "ca.kv.b", {(int64_t)(2 * c.d)});
    p.add(q + "ca.proj.w", {(int64_t)c.d, (int64_t)c.d});
    p.add(q + "ca.proj.b", {(int64_t)c.d});
    p.add(q + "ln3.w", {(int64_t)c.d});
    p.add(q + "ln3.b", {(int64_t)c.d});
    p.add(q + "ff1.w", {(int64_t)c.d, (int64_t)c.ff});
    p.add(q + "ff1.b", {(int64_t)c.ff});
    p.add(q + "ff2.w", {(int64_t)c.ff, (int64_t)c.d});
    p.add(q + "ff2.b", {(int64_t)c.d});
  }
  p.add("dec.lnf.w", {(int64_t)c.d});
  p.add("dec.lnf.b", {(int64_t)c.d});
  p.add("head.w", {(int64_t)c.d, (int64_t)V});
  p.add("head.b", {(int64_t)V});
}

// ---------------------------------------------------------------- 前向き

// 画像 [1,1,H,W]（0..1、1 = 黒）-> encoder の列 [S, d]
inline Tensor encode(const Tensor& img, const Params& p, const Cfg& c) {
  Tensor x = maxpool2d(relu(conv2d(img, p.get("enc.c1.w"), p.get("enc.c1.b"), 1, 1, 1)), 2, 2, 0);
  x = maxpool2d(relu(conv2d(x, p.get("enc.c2.w"), p.get("enc.c2.b"), 1, 1, 1)), 2, 2, 0);
  x = maxpool2d(relu(conv2d(x, p.get("enc.c3.w"), p.get("enc.c3.b"), 1, 1, 1)), 2, 2, 0);
  // [1, d, h, w] -> [h*w, d]。**転置してから並べる**（格子の順で列にする）
  const int64_t d = x->shape[1], h = x->shape[2], w = x->shape[3];
  return add(transpose2d(reshape(x, {d, h * w})), p.get("enc.pos"));
}

// multi-head attention。mask が空でなければ softmax の前に足す（因果マスク）
inline Tensor attn(const Tensor& q_in, const Tensor& kv_in, const Tensor& qw, const Tensor& qb,
                   const Tensor& kvw, const Tensor& kvb, const Tensor& pw, const Tensor& pb,
                   int heads, const Tensor& mask) {
  const int64_t d = q_in->shape[1], hd = d / heads;
  const Tensor q = add_rowvec(matmul(q_in, qw), qb);            // [T, d]
  const Tensor kv = add_rowvec(matmul(kv_in, kvw), kvb);        // [S, 2d]
  const float scale = 1.f / std::sqrt((float)hd);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    const Tensor qh = slice_cols(q, h * hd, h * hd + hd);
    const Tensor kh = slice_cols(kv, h * hd, h * hd + hd);
    const Tensor vh = slice_cols(kv, d + h * hd, d + h * hd + hd);
    Tensor s = mul_scalar(matmul(qh, transpose2d(kh)), scale);   // [T, S]
    if (mask) s = add(s, mask);
    outs.push_back(matmul(softmax_rows(s), vh));
  }
  return add_rowvec(matmul(hcat(outs), pw), pb);
}

// self-attention は qkv を 1 本で持つので、上とは別に書く（切り出す位置が違うだけ）
inline Tensor self_attn(const Tensor& x, const Tensor& qkvw, const Tensor& qkvb, const Tensor& pw,
                        const Tensor& pb, int heads, const Tensor& mask) {
  const int64_t d = x->shape[1], hd = d / heads;
  const Tensor qkv = add_rowvec(matmul(x, qkvw), qkvb);          // [T, 3d]
  const float scale = 1.f / std::sqrt((float)hd);
  std::vector<Tensor> outs;
  for (int h = 0; h < heads; ++h) {
    const Tensor q = slice_cols(qkv, h * hd, h * hd + hd);
    const Tensor k = slice_cols(qkv, d + h * hd, d + h * hd + hd);
    const Tensor v = slice_cols(qkv, 2 * d + h * hd, 2 * d + h * hd + hd);
    Tensor s = mul_scalar(matmul(q, transpose2d(k)), scale);
    if (mask) s = add(s, mask);
    outs.push_back(matmul(softmax_rows(s), v));
  }
  return add_rowvec(matmul(hcat(outs), pw), pb);
}

// decoder。ids は入力トークン（BOS から）。返すのは [T, V] のロジット
inline Tensor decode(const Tensor& enc, const std::vector<int>& ids, const Params& p,
                     const Cfg& c) {
  const int64_t T = (int64_t)ids.size();
  Tensor x = add(sq::embed_rows(p.get("dec.emb"), ids), slice_rows(p.get("dec.pos"), 0, T));
  const Tensor mask = sq::causal_mask(T);
  for (int l = 0; l < c.layers; ++l) {
    const std::string q = "dec." + std::to_string(l) + ".";
    const Tensor a = self_attn(ln(x, p.get(q + "ln1.w"), p.get(q + "ln1.b")),
                               p.get(q + "sa.qkv.w"), p.get(q + "sa.qkv.b"),
                               p.get(q + "sa.proj.w"), p.get(q + "sa.proj.b"), c.heads, mask);
    x = add(x, a);
    const Tensor b = attn(ln(x, p.get(q + "ln2.w"), p.get(q + "ln2.b")), enc,
                          p.get(q + "ca.q.w"), p.get(q + "ca.q.b"), p.get(q + "ca.kv.w"),
                          p.get(q + "ca.kv.b"), p.get(q + "ca.proj.w"), p.get(q + "ca.proj.b"),
                          c.heads, Tensor());
    x = add(x, b);
    Tensor h = add_rowvec(matmul(ln(x, p.get(q + "ln3.w"), p.get(q + "ln3.b")),
                                 p.get(q + "ff1.w")), p.get(q + "ff1.b"));
    h = add_rowvec(matmul(gelu(h), p.get(q + "ff2.w")), p.get(q + "ff2.b"));
    x = add(x, h);
  }
  x = ln(x, p.get("dec.lnf.w"), p.get("dec.lnf.b"));
  return add_rowvec(matmul(x, p.get("head.w")), p.get("head.b"));
}

// 画像を [1,1,H,W] の Tensor にする
inline Tensor img_tensor(const std::vector<float>& img, const Cfg& c) {
  Tensor t = make_tensor({1, 1, (int64_t)c.H, (int64_t)c.W}, false);
  t->data = img;
  return t;
}

// ---------------------------------------------------------------- 貪欲な生成
//
// **1 トークンずつ、そのたびに decoder を全部回す**（KV キャッシュは持たない）。
// max_len が 48 と短いので、素直に書いて速さより読みやすさを取る。
inline std::vector<int> greedy(const std::vector<float>& img, const Params& p, const Cfg& c,
                              int max_new = 0) {
  const int limit = max_new > 0 ? max_new : c.max_len - 1;
  const Tensor enc = encode(img_tensor(img, c), p, c);
  std::vector<int> ids{tok::BOS};
  for (int step = 0; step < limit; ++step) {
    const Tensor logits = decode(enc, ids, p, c);
    const int64_t T = logits->shape[0], V = logits->shape[1];
    int best = 0;
    float bp = -1e30f;
    for (int64_t v = 0; v < V; ++v) {
      const float s = logits->data[(T - 1) * V + v];
      if (s > bp) { bp = s; best = (int)v; }
    }
    free_graph(logits);
    if (best == tok::EOS) break;
    ids.push_back(best);
    if ((int)ids.size() >= c.max_len) break;
  }
  free_graph(enc);
  return ids;
}

// ---------------------------------------------------------------- 重みの読み書き（.pt）

inline void save(const Params& p, const std::string& path) {
  std::vector<pt::Tensor> ts;
  for (const std::string& n : p.order) {
    pt::Tensor t;
    t.name = n;
    t.shape = p.t.at(n)->shape;
    t.data = p.t.at(n)->data;
    ts.push_back(t);
  }
  pt::save_pt(ts, path);
}

inline bool load(Params& p, const std::string& path, std::string* why = nullptr) {
  std::vector<pt::Tensor> ts = pt::load_pt(path);
  if (ts.empty()) {
    if (why) *why = path + " が読めません";
    return false;
  }
  int hit = 0;
  for (const pt::Tensor& t : ts) {
    const std::map<std::string, Tensor>::iterator it = p.t.find(t.name);
    if (it == p.t.end()) continue;
    if ((int64_t)t.data.size() != it->second->numel()) {
      if (why) *why = t.name + " の大きさが違う";
      return false;
    }
    it->second->data = t.data;
    ++hit;
  }
  if (hit != (int)p.order.size()) {
    if (why) *why = "重みが足りません（" + std::to_string(hit) + " / " +
                    std::to_string(p.order.size()) + "）";
    return false;
  }
  return true;
}

}  // namespace mdl
