// Transformer に要る微分可能な演算 — LayerNorm、GELU(erf)、列ごとのスケール、
// 列の切り出しと連結（multi-head attention 用）、行の切り出し、ConvTranspose2d、双線形リサイズ。
//
// 出どころ: 姉妹リポ depth_anything_cpp/pure/depth_ops.hpp（ViT + DPT 用）。
// **中身はそのまま**で、この repo での役割が分かる名前に変えただけ。backward の登録の作法は
// autograd.hpp と同じ（行 = 先頭の次元の積、C = 最後の次元）。
#pragma once
#include "autograd.hpp"
#include "linalg.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>

// LayerNorm over the last dim.  y = gamma * (x-mean)/sqrt(var+eps) + beta
inline Tensor layernorm(const Tensor& x, const Tensor& g, const Tensor& b, float eps = 1e-6f) {
  int64_t C = x->shape.back(), R = x->numel() / C;
  auto o = make_tensor(x->shape, true);
  auto inv = std::make_shared<std::vector<float>>(R), mu = std::make_shared<std::vector<float>>(R);
  for (int64_t r = 0; r < R; ++r) { const float* xr = &x->data[r * C];
    double m = 0; for (int64_t c = 0; c < C; ++c) m += xr[c]; m /= C;
    double v = 0; for (int64_t c = 0; c < C; ++c) { double d = xr[c] - m; v += d * d; } v /= C;
    float iv = 1.f / std::sqrt((float)v + eps); (*mu)[r] = (float)m; (*inv)[r] = iv;
    float* yr = &o->data[r * C]; for (int64_t c = 0; c < C; ++c) yr[c] = g->data[c] * (xr[c] - (float)m) * iv + b->data[c]; }
  o->parents = {x, g, b}; Node* op = o.get();
  o->backward_fn = [x, g, b, op, R, C, inv, mu] {
    for (int64_t r = 0; r < R; ++r) { const float* xr = &x->data[r * C]; const float* gy = &op->grad[r * C];
      float iv = (*inv)[r], m = (*mu)[r]; double s1 = 0, s2 = 0;
      for (int64_t c = 0; c < C; ++c) { float xh = (xr[c] - m) * iv, dxh = gy[c] * g->data[c]; s1 += dxh; s2 += dxh * xh;
        g->grad[c] += gy[c] * xh; b->grad[c] += gy[c]; }
      for (int64_t c = 0; c < C; ++c) { float xh = (xr[c] - m) * iv, dxh = gy[c] * g->data[c];
        x->grad[r * C + c] += iv / C * (C * dxh - (float)s1 - xh * (float)s2); } } };
  return o;
}

// GELU (exact, erf):  0.5 x (1 + erf(x/√2))
inline Tensor gelu(const Tensor& x) {
  const float k = 0.70710678f, c0 = 0.39894228f;   // 1/√2, 1/√(2π)
  auto o = make_tensor(x->shape, true);
  for (int64_t i = 0; i < x->numel(); ++i) o->data[i] = 0.5f * x->data[i] * (1.f + std::erf(x->data[i] * k));
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, k, c0] { for (int64_t i = 0; i < x->numel(); ++i) { float v = x->data[i];
    float d = 0.5f * (1.f + std::erf(v * k)) + v * c0 * std::exp(-0.5f * v * v); x->grad[i] += d * op->grad[i]; } };
  return o;
}

// LayerScale / per-channel column scale:  y[r,c] = x[r,c] * g[c]
inline Tensor scale_cols(const Tensor& x, const Tensor& g) {
  int64_t C = x->shape.back(), R = x->numel() / C;
  auto o = make_tensor(x->shape, true);
  for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < C; ++c) o->data[r * C + c] = x->data[r * C + c] * g->data[c];
  o->parents = {x, g}; Node* op = o.get();
  o->backward_fn = [x, g, op, R, C] { for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < C; ++c) {
    x->grad[r * C + c] += op->grad[r * C + c] * g->data[c]; g->grad[c] += op->grad[r * C + c] * x->data[r * C + c]; } };
  return o;
}

// columns [c0,c1) of a 2D [R,C] tensor -> [R, c1-c0]
inline Tensor slice_cols(const Tensor& x, int64_t c0, int64_t c1) {
  int64_t C = x->shape.back(), R = x->numel() / C, W = c1 - c0;
  auto o = make_tensor({R, W}, true);
  for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < W; ++c) o->data[r * W + c] = x->data[r * C + c0 + c];
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, R, C, c0, W] { for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < W; ++c)
    x->grad[r * C + c0 + c] += op->grad[r * W + c]; };
  return o;
}
// horizontal concat of [R,Wi] blocks -> [R, sum Wi]
inline Tensor hcat(const std::vector<Tensor>& xs) {
  int64_t R = xs[0]->shape[0], W = 0; for (auto& t : xs) W += t->shape.back();
  auto o = make_tensor({R, W}, true);
  int64_t off = 0; for (auto& t : xs) { int64_t w = t->shape.back();
    for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < w; ++c) o->data[r * W + off + c] = t->data[r * w + c]; off += w; }
  o->parents = xs; Node* op = o.get();
  o->backward_fn = [xs, op, R, W] { int64_t off = 0; for (auto& t : xs) { int64_t w = t->shape.back();
    for (int64_t r = 0; r < R; ++r) for (int64_t c = 0; c < w; ++c) t->grad[r * w + c] += op->grad[r * W + off + c]; off += w; } };
  return o;
}

// vertical (row) concat of [Ri, C] tensors -> [sum Ri, C]  (e.g. prepend the cls token)
inline Tensor vcat(const std::vector<Tensor>& xs) {
  int64_t C = xs[0]->shape.back(), R = 0; for (auto& t : xs) R += t->numel() / C;
  auto o = make_tensor({R, C}, true); int64_t off = 0;
  for (auto& t : xs) { int64_t n = t->numel(); std::copy(t->data.begin(), t->data.end(), o->data.begin() + off); off += n; }
  o->parents = xs; Node* op = o.get();
  o->backward_fn = [xs, op] { int64_t off = 0; for (auto& t : xs) { int64_t n = t->numel();
    for (int64_t i = 0; i < n; ++i) t->grad[i] += op->grad[off + i]; off += n; } };
  return o;
}

// ConvTranspose2d (no groups): output (H-1)*s + k, no pad. w:[Cin,Cout,k,k], b:[Cout] or null.
inline Tensor convtranspose2d(const Tensor& x, const Tensor& w, const Tensor& b, int64_t stride) {
  int64_t N = x->shape[0], Cin = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t Cout = w->shape[1], k = w->shape[2], OH = (H - 1) * stride + k, OW = (W - 1) * stride + k;
  auto o = make_tensor({N, Cout, OH, OW}, true);
  if (b) for (int64_t n = 0; n < N; ++n) for (int64_t co = 0; co < Cout; ++co) for (int64_t p = 0; p < OH * OW; ++p) o->data[((n * Cout + co) * OH * OW) + p] = b->data[co];
  for (int64_t n = 0; n < N; ++n) for (int64_t ci = 0; ci < Cin; ++ci) for (int64_t ih = 0; ih < H; ++ih) for (int64_t iw = 0; iw < W; ++iw) {
    float xv = x->data[((n * Cin + ci) * H + ih) * W + iw];
    for (int64_t co = 0; co < Cout; ++co) for (int64_t kh = 0; kh < k; ++kh) for (int64_t kw = 0; kw < k; ++kw)
      o->data[((n * Cout + co) * OH + ih * stride + kh) * OW + iw * stride + kw] += xv * w->data[((ci * Cout + co) * k + kh) * k + kw]; }
  o->parents = {x, w}; if (b) o->parents.push_back(b); Node* op = o.get();
  o->backward_fn = [x, w, b, op, N, Cin, Cout, H, W, k, OH, OW, stride] {
    for (int64_t n = 0; n < N; ++n) for (int64_t ci = 0; ci < Cin; ++ci) for (int64_t ih = 0; ih < H; ++ih) for (int64_t iw = 0; iw < W; ++iw) {
      float xv = x->data[((n * Cin + ci) * H + ih) * W + iw], gx = 0;
      for (int64_t co = 0; co < Cout; ++co) for (int64_t kh = 0; kh < k; ++kh) for (int64_t kw = 0; kw < k; ++kw) {
        float go = op->grad[((n * Cout + co) * OH + ih * stride + kh) * OW + iw * stride + kw];
        gx += go * w->data[((ci * Cout + co) * k + kh) * k + kw]; w->grad[((ci * Cout + co) * k + kh) * k + kw] += go * xv; }
      x->grad[((n * Cin + ci) * H + ih) * W + iw] += gx; }
    if (b) for (int64_t n = 0; n < N; ++n) for (int64_t co = 0; co < Cout; ++co) { float s = 0;
      for (int64_t p = 0; p < OH * OW; ++p) s += op->grad[(n * Cout + co) * OH * OW + p]; b->grad[co] += s; } };
  return o;
}

// rows [r0,r1) of a 2D [R,C] tensor -> [r1-r0, C]  (e.g. drop the cls token: slice_rows(x,1,T))
inline Tensor slice_rows(const Tensor& x, int64_t r0, int64_t r1) {
  int64_t C = x->shape.back(); auto o = make_tensor({r1 - r0, C}, true);
  std::copy(x->data.begin() + r0 * C, x->data.begin() + r1 * C, o->data.begin());
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, r0, C] { for (int64_t i = 0; i < op->numel(); ++i) x->grad[r0 * C + i] += op->grad[i]; };
  return o;
}

// bilinear resize NCHW -> (N,C,OH,OW). align_corners: false = PyTorch/ONNX default; true = DPT fusion.
inline Tensor resize_bilinear(const Tensor& x, int64_t OH, int64_t OW, bool align_corners = false) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  auto o = make_tensor({N, C, OH, OW}, true); float sh = (float)H / OH, sw = (float)W / OW;
  float ah = (OH > 1) ? (float)(H - 1) / (OH - 1) : 0.f, aw = (OW > 1) ? (float)(W - 1) / (OW - 1) : 0.f;
  auto idx = std::make_shared<std::vector<std::array<float, 6>>>(OH * OW);   // y0,y1,fy, x0,x1,fx
  for (int64_t oy = 0; oy < OH; ++oy) for (int64_t ox = 0; ox < OW; ++ox) {
    float sy = align_corners ? oy * ah : (oy + 0.5f) * sh - 0.5f, sx = align_corners ? ox * aw : (ox + 0.5f) * sw - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    (*idx)[oy * OW + ox] = {(float)std::clamp(y0, 0, (int)H - 1), (float)std::clamp(y0 + 1, 0, (int)H - 1), fy,
                            (float)std::clamp(x0, 0, (int)W - 1), (float)std::clamp(x0 + 1, 0, (int)W - 1), fx}; }
  for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) { const float* xp = &x->data[(n * C + c) * H * W]; float* yp = &o->data[(n * C + c) * OH * OW];
    for (int64_t p = 0; p < OH * OW; ++p) { auto& q = (*idx)[p]; int y0 = (int)q[0], y1 = (int)q[1], x0 = (int)q[3], x1 = (int)q[4]; float fy = q[2], fx = q[5];
      yp[p] = xp[y0 * W + x0] * (1 - fx) * (1 - fy) + xp[y0 * W + x1] * fx * (1 - fy) + xp[y1 * W + x0] * (1 - fx) * fy + xp[y1 * W + x1] * fx * fy; } }
  o->parents = {x}; Node* op = o.get();
  o->backward_fn = [x, op, N, C, H, W, OH, OW, idx] { for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) {
    float* gx = &x->grad[(n * C + c) * H * W]; const float* gy = &op->grad[(n * C + c) * OH * OW];
    for (int64_t p = 0; p < OH * OW; ++p) { auto& q = (*idx)[p]; int y0 = (int)q[0], y1 = (int)q[1], x0 = (int)q[3], x1 = (int)q[4]; float fy = q[2], fx = q[5], g = gy[p];
      gx[y0 * W + x0] += g * (1 - fx) * (1 - fy); gx[y0 * W + x1] += g * fx * (1 - fy); gx[y1 * W + x0] += g * (1 - fx) * fy; gx[y1 * W + x1] += g * fx * fy; } } };
  return o;
}
