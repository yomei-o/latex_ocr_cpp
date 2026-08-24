// pix2tex に入れる画像を作る（本家 utils.pad + cli.minmax_size と同じ手順）。
//
// **前処理は精度をほぼ決める。** 同じ式でも字の大きさが学習時と違うと本家でも読めない
// （実測: image_resizer 無しだと "E = mc^2" が \underline{{{\cal I}}} になり、有りだと E=m c^{2}）。
// だから手順は本家に合わせる。合わせた結果、tools/pix2tex.py（PIL を使う）と
// この実装（自前）で、6 枚の試験画像すべて同じ大きさ・同じ LaTeX になることを確かめてある。
//
// 縮小は PIL と同じ係数の作り方にしてある（Lanczos は support=3 を縮小率で広げる）。
// 「面積平均」や「間引き」で代用すると、細い線が消えて読みが変わる。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ip {

struct Img {                                    // 8bit グレイ。0 = 黒
  int w = 0, h = 0;
  std::vector<unsigned char> p;
  unsigned char at(int x, int y) const { return p[(size_t)y * w + x]; }
  void set(int x, int y, unsigned char v) { p[(size_t)y * w + x] = v; }
  bool empty() const { return w <= 0 || h <= 0; }
};

inline Img make(int w, int h, unsigned char fill = 255) {
  Img im;
  im.w = w;
  im.h = h;
  im.p.assign((size_t)w * h, fill);
  return im;
}

// ---------------------------------------------------------------- 拡大縮小（PIL と同じ係数）

enum Filter { BILINEAR, LANCZOS };

inline double filt(Filter f, double x) {
  x = std::fabs(x);
  if (f == BILINEAR) return x < 1.0 ? 1.0 - x : 0.0;
  if (x >= 3.0) return 0.0;
  if (x < 1e-8) return 1.0;
  const double px = 3.14159265358979323846 * x;
  return std::sin(px) / px * std::sin(px / 3.0) / (px / 3.0);
}

// 1 方向ぶんの係数（PIL の precompute_coeffs と同じ）
inline void coeffs(int in_size, int out_size, Filter f, std::vector<std::vector<double>>& ws,
                   std::vector<int>& mins) {
  const double scale = (double)in_size / (double)out_size;
  const double fscale = std::max(1.0, scale);
  const double support = (f == BILINEAR ? 1.0 : 3.0) * fscale;
  const double ss = 1.0 / fscale;
  ws.assign((size_t)out_size, {});
  mins.assign((size_t)out_size, 0);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    int xmin = (int)(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    std::vector<double> k;
    double sum = 0;
    for (int x = xmin; x < xmax; ++x) {
      const double w = filt(f, (x - center + 0.5) * ss);
      k.push_back(w);
      sum += w;
    }
    if (sum != 0)
      for (double& w : k) w /= sum;
    ws[(size_t)xx] = k;
    mins[(size_t)xx] = xmin;
  }
}

inline unsigned char clamp8(double v) {
  const int i = (int)std::floor(v + 0.5);
  return (unsigned char)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

inline Img resize(const Img& src, int W, int H, Filter f) {
  if (src.w == W && src.h == H) return src;
  std::vector<std::vector<double>> kw, kh;
  std::vector<int> mw, mh;
  coeffs(src.w, W, f, kw, mw);
  coeffs(src.h, H, f, kh, mh);
  Img mid = make(W, src.h);                     // 横 -> 縦（PIL と同じ順。間に 8bit を挟む）
  for (int y = 0; y < src.h; ++y)
    for (int x = 0; x < W; ++x) {
      double acc = 0;
      for (size_t k = 0; k < kw[(size_t)x].size(); ++k)
        acc += kw[(size_t)x][k] * src.at(mw[(size_t)x] + (int)k, y);
      mid.set(x, y, clamp8(acc));
    }
  Img out = make(W, H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      double acc = 0;
      for (size_t k = 0; k < kh[(size_t)y].size(); ++k)
        acc += kh[(size_t)y][k] * mid.at(x, mh[(size_t)y] + (int)k);
      out.set(x, y, clamp8(acc));
    }
  return out;
}

// ---------------------------------------------------------------- 本家の 2 つの部品

// 明暗を目一杯に伸ばし、白地に黒字にして、インクのある所を切り出し、
// 32 の倍数に切り上げて白で埋める（画像は左上）。本家 utils.pad。
inline Img pil_pad(const Img& in, int divable = 32) {
  if (in.empty()) return in;
  unsigned char lo = 255, hi = 0;
  for (unsigned char v : in.p) { lo = std::min(lo, v); hi = std::max(hi, v); }
  std::vector<float> a((size_t)in.w * in.h);
  double mean = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = hi > lo ? (float)((in.p[i] - lo) * 255.0 / (hi - lo)) : 255.f;
    mean += a[i];
  }
  mean /= (double)a.size();
  const bool light = mean > 128.0;              // 明るい = 白地に黒字
  if (!light)
    for (float& v : a) v = 255.f - v;           // 黒地に白字 -> 反転
  int x0 = in.w, y0 = in.h, x1 = -1, y1 = -1;
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      const float v = a[(size_t)y * in.w + x];
      const bool ink = light ? (v < 128.f) : (255.f - v > 128.f);
      if (ink) {
        x0 = std::min(x0, x); y0 = std::min(y0, y);
        x1 = std::max(x1, x); y1 = std::max(y1, y);
      }
    }
  if (x1 < 0) { x0 = y0 = 0; x1 = in.w - 1; y1 = in.h - 1; }
  const int cw = x1 - x0 + 1, ch = y1 - y0 + 1;
  const int W = (cw + divable - 1) / divable * divable;
  const int H = (ch + divable - 1) / divable * divable;
  Img out = make(W, H);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      out.set(x, y, clamp8(a[(size_t)(y0 + y) * in.w + (x0 + x)]));
  return out;
}

// 大きすぎたら縮める、小さすぎたら白で足す。本家 cli.minmax_size。
inline Img minmax(const Img& in, int max_w, int max_h, int min_w, int min_h) {
  Img im = in;
  const double r = std::max((double)im.w / max_w, (double)im.h / max_h);
  if (r > 1.0) im = resize(im, (int)(im.w / r), (int)(im.h / r), BILINEAR);
  const int W = std::max(im.w, min_w), H = std::max(im.h, min_h);
  if (W != im.w || H != im.h) {
    Img o = make(W, H);
    for (int y = 0; y < im.h; ++y)
      for (int x = 0; x < im.w; ++x) o.set(x, y, im.at(x, y));
    im = o;
  }
  return im;
}

}  // namespace ip
