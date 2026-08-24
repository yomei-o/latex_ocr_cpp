// 列を扱うのに要る微分可能な演算 — 埋め込みの引き出し、因果マスク、交差エントロピー。
//
// engine（autograd.hpp）には conv も matmul も softmax もあるが、**行をばらばらに拾う**
// 演算だけ無い。埋め込みは「表から id の行を持ってくる」なので、それを 1 つ足す。
// backward は拾った行に足し戻すだけ（同じ id が何回出ても足し合わされる）。
#pragma once
#include "autograd.hpp"
#include "ops2d.hpp"
#include <cmath>
#include <vector>

namespace sq {

// 表 [V, D] から ids の行を並べて [T, D] を作る（埋め込み）
inline Tensor embed_rows(const Tensor& table, const std::vector<int>& ids) {
  const int64_t D = table->shape[1];
  const int64_t T = (int64_t)ids.size();
  Tensor o = make_tensor({T, D}, true);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t r = ids[(size_t)t];
    for (int64_t d = 0; d < D; ++d) o->data[t * D + d] = table->data[r * D + d];
  }
  o->parents = {table};
  Node* op = o.get();
  o->backward_fn = [table, op, ids, T, D] {
    for (int64_t t = 0; t < T; ++t) {
      const int64_t r = ids[(size_t)t];
      for (int64_t d = 0; d < D; ++d) table->grad[r * D + d] += op->grad[t * D + d];
    }
  };
  return o;
}

// 因果マスク [T, T]。**定数**（勾配は要らない）。0 か -1e9 で、softmax の前に足す
inline Tensor causal_mask(int64_t T) {
  Tensor m = make_tensor({T, T}, false);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < T; ++j) m->data[i * T + j] = (j <= i) ? 0.f : -1e9f;
  return m;
}

// 行ごとの重み付き和 -> スカラ（PAD を外して平均を取るのに使う）
inline Tensor masked_mean(const Tensor& a, const std::vector<float>& w) {
  const int64_t N = a->numel();
  Tensor o = make_tensor({1, 1}, true);
  double s = 0, ws = 0;
  for (int64_t i = 0; i < N; ++i) { s += (double)a->data[i] * w[(size_t)i]; ws += w[(size_t)i]; }
  const float inv = ws > 0 ? (float)(1.0 / ws) : 0.f;
  o->data[0] = (float)(s * inv);
  o->parents = {a};
  Node* op = o.get();
  o->backward_fn = [a, op, w, N, inv] {
    for (int64_t i = 0; i < N; ++i) a->grad[i] += op->grad[0] * w[(size_t)i] * inv;
  };
  return o;
}

// 交差エントロピー。logits [T, V]、targets は長さ T、mask は 1 か 0（PAD を外す）
inline Tensor ce_loss(const Tensor& logits, const std::vector<int>& targets,
                      const std::vector<float>& mask) {
  const Tensor lp = log_softmax_rows(logits);
  std::vector<int64_t> idx;
  idx.reserve(targets.size());
  for (int t : targets) idx.push_back((int64_t)t);
  const Tensor picked = gather_row(lp, idx);           // [T, 1]
  return mul_scalar(masked_mean(picked, mask), -1.f);
}

}  // namespace sq
