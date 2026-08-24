// 学習データ — 数式の画像と、その LaTeX。
//
// **データは自分で作る。** 外からデータセットを落としてこないのがこの repo の作り方で、
// 理由は 2 つ:
//   * 画像と LaTeX が**必ず一致する**（人手の注釈と違って、ずれようがない）
//   * いくらでも作れるので、「データが足りない」と「モデルが弱い」を混同しないで済む
//
// 作り方: 乱数で式を作る（gen_expr）-> 式木を組版して画像にする（typeset）->
// 同じ式木から LaTeX を書く（ex::to_latex）。**同じ木から絵と答えを作る**のが要点で、
// ここが別々だと絵と答えが食い違う（LaTeX-OCR で一番まずい壊れ方）。
//
// 画像は決まった大きさの帯（既定 64x256、白地に黒字）に収める。長い式は縮めて入れ、
// 縦横比は保つ。モデルの入力はこの帯そのもの。
#pragma once
#include "expr.hpp"
#include "gen_expr.hpp"
#include "rng.hpp"
#include "tok.hpp"
#include "typeset.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace dat {

struct Cfg {
  int W = 256, H = 64;        // 入力の帯
  int px_min = 24, px_max = 40;
  int max_len = 48;           // トークン列の上限（BOS と EOS を含む）
  int margin = 4;
};

struct Sample {
  std::vector<float> img;     // [H*W] 0..1（1 = 黒）。モデルに入れる向き
  std::vector<int> ids;       // BOS ... EOS（max_len まで PAD で埋める）
  int len = 0;                // PAD を除いた長さ
  std::string latex;
};

// 描いた絵を帯に収める。**縦横比は保つ**（潰すと字の形が変わって、実物と違う絵で学習する）
inline void fit_into(const ts::Rendered& r, const Cfg& c, std::vector<float>& out) {
  out.assign((size_t)c.H * c.W, 0.f);
  if (r.w <= 0 || r.h <= 0) return;
  const int aw = c.W - 2 * c.margin, ah = c.H - 2 * c.margin;
  double s = std::min((double)aw / r.w, (double)ah / r.h);
  if (s > 1.0) s = 1.0;                                // 小さい式は拡大しない（線が太る）
  const int dw = std::max(1, (int)(r.w * s)), dh = std::max(1, (int)(r.h * s));
  const int ox = (c.W - dw) / 2, oy = (c.H - dh) / 2;
  for (int y = 0; y < dh; ++y)
    for (int x = 0; x < dw; ++x) {
      // 面積平均で縮める（間引くと細い線が消える。分数線は 2 画素しかない）
      const int sx0 = (int)(x / s), sx1 = std::max(sx0 + 1, (int)((x + 1) / s));
      const int sy0 = (int)(y / s), sy1 = std::max(sy0 + 1, (int)((y + 1) / s));
      double acc = 0;
      int n = 0;
      for (int yy = sy0; yy < std::min(sy1, r.h); ++yy)
        for (int xx = sx0; xx < std::min(sx1, r.w); ++xx) {
          acc += 255.0 - r.gray[(size_t)yy * r.w + xx];   // 白地黒字 -> インクの量
          ++n;
        }
      if (n) out[(size_t)(oy + y) * c.W + (ox + x)] = (float)(acc / n / 255.0);
    }
}

// 式を 1 つ組んで帯に収める（gen_expr が出すのと同じ記法。`frac(1,2) + sqrt(x)`）。
// latex にはその式の LaTeX が入る。**画像と LaTeX は同じ木から作る**ので食い違わない。
// 学習データは make_one が作るが、絵の作り方（fit_into）はこちらと同じ。
inline bool render_src(const ts::Font& font, const ts::Font* font_i, const ts::Style& st,
                       const std::string& src, const Cfg& c, int px, std::vector<float>& img,
                       std::string& latex) {
  std::string why;
  const ex::E e = ex::parse(src, &why);
  if (!why.empty()) return false;
  latex = ex::to_latex(e);
  const ts::Rendered r = ts::render(font, font_i, e, px, st);
  if (r.w <= 0 || r.h <= 0) return false;
  fit_into(r, c, img);
  return true;
}

// 1 件作る。作れなければ ok=false（式が解析できない・トークンが語彙に無い・長すぎる）
inline bool make_one(const ts::Font& font, const ts::Font* font_i, const ts::Style& st, Rng& rng,
                     const Cfg& c, Sample& out) {
  std::string why;
  const std::string src = gx::one(rng);
  const ex::E e = ex::parse(src, &why);
  if (!why.empty()) return false;
  out.latex = ex::to_latex(e);
  const std::vector<int> ids = tok::encode(out.latex, true, true);
  if ((int)ids.size() > c.max_len) return false;
  for (int id : ids)
    if (id == tok::UNK) return false;                  // 語彙に無い字が出たら捨てる（黙って学習しない）
  const int px = c.px_min + (int)rng.below((uint64_t)(c.px_max - c.px_min + 1));
  const ts::Rendered r = ts::render(font, font_i, e, px, st);
  if (r.w <= 0 || r.h <= 0) return false;
  fit_into(r, c, out.img);
  out.ids = ids;
  out.len = (int)ids.size();
  out.ids.resize((size_t)c.max_len, tok::PAD);
  return true;
}

}  // namespace dat
