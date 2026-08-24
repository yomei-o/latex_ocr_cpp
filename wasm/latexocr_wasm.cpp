// ブラウザ側の入口。pure/ の同じコードを emscripten で固める。
//
// **学習した本体をそのまま持ってくる**（推論用に書き直したものではない）。だから
// ブラウザで出る文字列は latexocr.exe infer と同じで、片方だけ直すことが起きない。
//
// フォントと重みは --preload-file で MEMFS に入れてあるので、pure/ 側の
// 「パスから読む」API をそのまま使える（wasm 用の読み込み口を別に作らない）。
//
//   sh build/emcc.sh
#define STB_IMAGE_IMPLEMENTATION
#define STB_TRUETYPE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#include "third_party/stb_image.h"
#include "typeset_impl.hpp"
#include "data.hpp"
#include "model.hpp"
#include "tok.hpp"
#include <emscripten/emscripten.h>
#include <string>
#include <vector>

namespace {

ts::Font g_font;
ts::Style g_style;
mdl::Cfg g_cfg;
mdl::Params g_params;
bool g_ready = false;
std::string g_out;                       // JS に返す文字列の置き場（返した先で使い終わるまで持つ）

dat::Cfg data_cfg() {
  dat::Cfg dc;
  dc.W = g_cfg.W;
  dc.H = g_cfg.H;
  dc.max_len = g_cfg.max_len;
  return dc;
}

}  // namespace

extern "C" {

// 画像の大きさ（JS 側で canvas を作るのに要る）
EMSCRIPTEN_KEEPALIVE int lx_w() { return g_cfg.W; }
EMSCRIPTEN_KEEPALIVE int lx_h() { return g_cfg.H; }
EMSCRIPTEN_KEEPALIVE int lx_ready() { return g_ready ? 1 : 0; }

// フォントと重みを読む。0 なら成功、それ以外は失敗（理由は lx_why）。
EMSCRIPTEN_KEEPALIVE int lx_init(const char* font_path, const char* model_path) {
  std::string why;
  if (!ts::load_font(g_font, font_path, &why)) { g_out = why; return 1; }
  g_style.italic_vars = false;
  g_cfg.vocab = tok::size();
  mdl::build(g_params, g_cfg);
  if (!mdl::load(g_params, model_path, &why)) { g_out = why; return 2; }
  g_ready = true;
  return 0;
}

EMSCRIPTEN_KEEPALIVE const char* lx_why() { return g_out.c_str(); }

// 式（`frac(1,2) + sqrt(x)` の書き方）を組んで W*H のグレイ画像にする（0 = 黒、255 = 白）。
// 組めなければ 0。組めたら、その式の LaTeX が lx_why で取れる。
EMSCRIPTEN_KEEPALIVE int lx_render(const char* src, int px, unsigned char* out) {
  if (!g_ready) return 0;
  std::vector<float> img;
  std::string latex;
  if (!dat::render_src(g_font, nullptr, g_style, src, data_cfg(), px, img, latex)) {
    g_out = "式として読めません";
    return 0;
  }
  for (size_t i = 0; i < img.size(); ++i)
    out[i] = (unsigned char)(255.f - 255.f * img[i]);
  // **トークンに割って戻した形**を返す（生の LaTeX と読みを比べると、空白の入り方が
  // 違うだけで「外れ」に見える。モデルが出せるのはトークン列なので、そちらに揃える）
  g_out = tok::decode(tok::encode(latex));
  return 1;
}

// でたらめな式を 1 つ作って、画像と正解の LaTeX を返す（正解は lx_why で取る）。
EMSCRIPTEN_KEEPALIVE int lx_sample(unsigned int seed, unsigned char* out) {
  if (!g_ready) return 0;
  Rng rng(seed);
  const dat::Cfg dc = data_cfg();
  dat::Sample s;
  for (int i = 0; i < 40; ++i) {
    if (!dat::make_one(g_font, nullptr, g_style, rng, dc, s)) continue;
    for (size_t k = 0; k < s.img.size(); ++k)
      out[k] = (unsigned char)(255.f - 255.f * s.img[k]);
    g_out = tok::decode(tok::encode(s.latex));
    return 1;
  }
  return 0;
}

// 好きな大きさのグレイ画像（0 = 黒、白地）を、学習時と**同じやり方で**帯に収める。
// JS 側で縮めると縮め方が違ってしまう（学習は面積平均で縮め、拡大はしない）ので、
// ここでやる。out は W*H。
EMSCRIPTEN_KEEPALIVE int lx_fit(const unsigned char* gray, int w, int h, unsigned char* out) {
  if (!g_ready || w <= 0 || h <= 0) return 0;
  ts::Rendered r;
  r.w = w;
  r.h = h;
  r.gray.assign(gray, gray + (size_t)w * h);
  std::vector<float> img;
  dat::fit_into(r, data_cfg(), img);
  for (size_t i = 0; i < img.size(); ++i)
    out[i] = (unsigned char)(255.f - 255.f * img[i]);
  return 1;
}

// W*H のグレイ画像（0 = 黒）を読んで LaTeX を返す。
EMSCRIPTEN_KEEPALIVE const char* lx_read(const unsigned char* gray) {
  static std::string r;
  if (!g_ready) { r = ""; return r.c_str(); }
  std::vector<float> img((size_t)g_cfg.H * g_cfg.W);
  for (size_t i = 0; i < img.size(); ++i) img[i] = (255.f - gray[i]) / 255.f;
  r = tok::decode(mdl::greedy(img, g_params, g_cfg));
  return r.c_str();
}

}  // extern "C"
