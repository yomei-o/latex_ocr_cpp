// 組版の実装のうち、stb_truetype の実体が必要な部分。**.cpp からだけ** include する
// （stb は実装がインクルードガードの外にあるので、ヘッダから include すると二重定義になる。
// 姉妹リポにも同じ注意書きがある）。
#pragma once
#include "stb_truetype.h"
#include "typeset.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ts {

// **画像で持つ字が優先。** 寸法は書き出し元の単位なので、この書体の単位に直す
// （両言語とも同じ整数演算で直すので、枠は 1px も違わない）。
inline int Font::advance(int cp) const {
  const std::map<int, BitGlyph>::const_iterator it = bits.find(cp);
  if (bit_upem > 0 && it != bits.end())
    return (int)((long long)it->second.adv * upem / bit_upem);
  int adv = 0, lsb = 0;
  stbtt_GetCodepointHMetrics(info, cp, &adv, &lsb);
  return adv;
}

inline void Font::bbox(int cp, int* x0, int* y0, int* x1, int* y1) const {
  const std::map<int, BitGlyph>::const_iterator it = bits.find(cp);
  if (bit_upem > 0 && it != bits.end()) {
    *x0 = (int)((long long)it->second.x0 * upem / bit_upem);
    *y0 = (int)((long long)it->second.y0 * upem / bit_upem);
    *x1 = (int)((long long)it->second.x1 * upem / bit_upem);
    *y1 = (int)((long long)it->second.y1 * upem / bit_upem);
    return;
  }
  *x0 = *y0 = *x1 = *y1 = 0;
  stbtt_GetCodepointBox(info, cp, x0, y0, x1, y1);
}

// フォントを読む。--font で渡されなければ、よくある場所を順に探す。
// **フォントはリポジトリに入れない**（再配布の可否が font ごとに違う）。学習データを作る側が
// 手元のフォントを指定する。Kaggle（Linux）には DejaVu がある。
inline bool load_font(Font& f, const std::string& path_in, std::string* why = nullptr,
                      bool italic = false) {
  static const char* kRoman[] = {
      "fonts/math.ttf",
      "C:/Windows/Fonts/times.ttf",
      "C:/Windows/Fonts/georgia.ttf",
      "C:/Windows/Fonts/arial.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
  };
  // 数式のイタリック体。**教科書の変数はこれで組まれている**。Kaggle（Linux）には
  // matplotlib が Computer Modern（cmmi10）と STIX を同梱しているので、そこも見る
  // （cmmi10 は TeX の数式イタリックそのもので、教科書の字形に最も近い）。
  static const char* kItalic[] = {
      "fonts/math-italic.ttf",
      "C:/Windows/Fonts/timesi.ttf",
      "C:/Windows/Fonts/cambriai.ttf",
      "C:/Windows/Fonts/georgiai.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
  };
  std::vector<std::string> tries;
  if (!path_in.empty()) tries.push_back(path_in);
  if (italic) { for (const char* c : kItalic) tries.push_back(c); }
  else { for (const char* c : kRoman) tries.push_back(c); }

  for (const std::string& p : tries) {
    FILE* fp = fopen(p.c_str(), "rb");
    if (!fp) continue;
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    f.data.assign((size_t)n, 0);
    const size_t got = fread(f.data.data(), 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) continue;
    f.info = new stbtt_fontinfo();
    const int off = stbtt_GetFontOffsetForIndex(f.data.data(), 0);
    if (!stbtt_InitFont(f.info, f.data.data(), off < 0 ? 0 : off)) { delete f.info; f.info = nullptr; continue; }
    // upem は head テーブルの値。stb は直接返さないので、'M' の縦幅ではなく
    // unitsPerEm を自分で読む（TTF のバイト列から: head テーブルの offset+18）
    f.upem = 0;
    {
      const unsigned char* d = f.data.data();
      // **.ttc は先頭が ttcf で、表の目録は先頭ではなく off から始まる。**
      // ここを見ていなかったので、教科書体（.ttc）の upem が 1000 に落ちて、
      // 字の寸法が 2 倍ずれていた（fontdump で気付いた）。
      const unsigned char* base = d + (off < 0 ? 0 : off);
      const int num = (base[4] << 8) | base[5];
      for (int i = 0; i < num; ++i) {
        const unsigned char* rec = base + 12 + 16 * i;
        if (memcmp(rec, "head", 4) == 0) {
          const unsigned int o = (rec[8] << 24) | (rec[9] << 16) | (rec[10] << 8) | rec[11];
          f.upem = (d[o + 18] << 8) | d[o + 19];
          break;
        }
      }
    }
    if (f.upem <= 0) f.upem = 1000;
    int lg = 0;
    stbtt_GetFontVMetrics(f.info, &f.ascent, &f.descent, &lg);
    return true;
  }
  if (why) *why = "フォントが見つかりません（--font で TTF を渡してください）";
  return false;
}

// 字を 1 つ、w x h の被覆率に焼く（bbox にぴったり収める）。`mathcam fontdump` が使う。
inline void rasterize_glyph(const Font& f, int cp, unsigned char* out, int w, int h) {
  std::memset(out, 0, (size_t)w * h);
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  f.bbox(cp, &x0, &y0, &x1, &y1);
  if (x1 <= x0 || y1 <= y0 || w <= 0 || h <= 0) return;
  // bbox がちょうど w x h になる倍率で焼く（縦横で別々に取ると字が歪むので、面で合わせる）
  const float sx = (float)w / (float)(x1 - x0), sy = (float)h / (float)(y1 - y0);
  stbtt_MakeCodepointBitmapSubpixel(f.info, out, w, h, w, sx, sy, 0.f, 0.f, cp);
}

// 画像で持つ字を読む（`mathcam fontdump` が書いたディレクトリ）。
//
// metrics.txt の書き方（**両言語がこの整数をそのまま読む**ので、枠が一致する）:
//   upem <n>
//   <符号位置> <advance> <x0> <y0> <x1> <y1> <画像ファイル名> <画像の幅> <画像の高さ>
// 画像は 8bit グレースケールで、**bbox にぴったり**。値は被覆率（255 で真っ黒）。
inline bool load_bitmap_glyphs(Font& f, const std::string& dir, std::string* why = nullptr) {
  const std::string mp = dir + "/metrics.txt";
  FILE* fp = fopen(mp.c_str(), "rb");
  if (!fp) { if (why) *why = "画像の字が見つかりません: " + mp; return false; }
  char line[512];
  int upem = 0;
  int loaded = 0;
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    int cp = 0, adv = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0, w = 0, h = 0;
    char name[128] = {0};
    if (sscanf(line, "upem %d", &upem) == 1) continue;
    if (sscanf(line, "%d %d %d %d %d %d %127s %d %d", &cp, &adv, &x0, &y0, &x1, &y1, name,
               &w, &h) != 9)
      continue;
    BitGlyph g;
    g.adv = adv; g.x0 = x0; g.y0 = y0; g.x1 = x1; g.y1 = y1; g.w = w; g.h = h;
    if (w > 0 && h > 0) {
      int iw = 0, ih = 0, ic = 0;
      unsigned char* px = stbi_load((dir + "/" + name).c_str(), &iw, &ih, &ic, 1);
      if (!px || iw != w || ih != h) {
        if (px) stbi_image_free(px);
        fclose(fp);
        if (why) *why = "画像の字が読めません: " + dir + "/" + name;
        return false;
      }
      g.pix.assign(px, px + (size_t)w * h);
      stbi_image_free(px);
    }
    f.bits[cp] = g;
    ++loaded;
  }
  fclose(fp);
  if (upem <= 0 || loaded == 0) {
    if (why) *why = "metrics.txt が読めません: " + mp;
    return false;
  }
  f.bit_upem = upem;
  return true;
}

// 組版して画素に落とす。返る箱は「画素・y は下向き正」で、認識器の正解枠にそのまま使える。
inline Rendered render_p(const Font& f, const Font* fi, const P& p, int px,
                         const Style& st = Style()) {
  Rendered R;
  Layout L = lay(f, fi, p, 1, 1);

  // 実際に置かれたものの範囲を取る（layout の box より、描いた枠の合併のほうが正確）
  int minx = 0, maxx = L.box.w, miny = -L.box.desc, maxy = L.box.asc;
  for (const Item& it : L.items) {
    // bbox は shift() で既に絶対座標になっている。ここで it.x を足すと二重加算になる
    // （実際にそう書いていて、ラベルの枠だけが 2 倍の位置に出ていた。絵は it.x を使う
    //  ラスタ側だけ正しかったので、見た目では気付けなかった）。
    const int bx0 = it.x0, by0 = it.y0, bx1 = it.x1, by1 = it.y1;
    minx = std::min(minx, bx0); maxx = std::max(maxx, bx1);
    miny = std::min(miny, by0); maxy = std::max(maxy, by1);
  }
  const int mg = emk(f, K_MARGIN);
  const int W = to_px((long long)(maxx - minx) + 2 * mg, px, f.upem);
  const int H = to_px((long long)(maxy - miny) + 2 * mg, px, f.upem);
  R.w = std::max(1, W);
  R.h = std::max(1, H);
  R.gray.assign((size_t)R.w * R.h, (unsigned char)st.paper);

  const int ox = -minx + mg;                        // フォント単位での平行移動
  const int oy = maxy + mg;                         // 上端をこの位置に置く（y は上向き正のまま）

  for (const Item& it : L.items) {
    if (it.cp == 0) {                               // 線（分数・根号の横線）
      const int x0 = to_px((long long)it.x0 + ox, px, f.upem);
      const int x1 = to_px((long long)it.x1 + ox, px, f.upem);
      const int y0 = to_px((long long)oy - it.y1, px, f.upem);   // 上向き -> 下向き
      const int y1 = to_px((long long)oy - it.y0, px, f.upem);
      for (int y = std::max(0, y0); y < std::min(R.h, std::max(y1, y0 + 1)); ++y)
        for (int x = std::max(0, x0); x < std::min(R.w, x1); ++x)
          R.gray[(size_t)y * R.w + x] = (unsigned char)st.ink;
      R.cls.push_back(it.cls);
      R.box.push_back(x0); R.box.push_back(y0); R.box.push_back(x1);
      R.box.push_back(std::max(y1, y0 + 1));
      continue;
    }
    // 記号。stb に「その文字だけ」のビットマップを描かせて貼る
    const Font& g = (it.ital && fi) ? *fi : f;
    const float sc = (float)px / (float)g.upem * (float)it.scale_num / (float)it.scale_den;
    int gx0 = 0, gy0 = 0, gx1 = 0, gy1 = 0;
    if (g.has_bit(it.cp)) {
      // 画像で持つ字。stb と同じ丸め方で画素の箱を出す（x0,y0 は floor、x1,y1 は ceil）
      int bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;
      g.bbox(it.cp, &bx0, &by0, &bx1, &by1);
      gx0 = (int)std::floor(bx0 * sc);
      gy0 = (int)std::floor(-by1 * sc);
      gx1 = (int)std::ceil(bx1 * sc);
      gy1 = (int)std::ceil(-by0 * sc);
    } else {
      stbtt_GetCodepointBitmapBox(g.info, it.cp, sc, sc, &gx0, &gy0, &gx1, &gy1);
    }
    const int gw = gx1 - gx0, gh = gy1 - gy0;
    const int penx = to_px((long long)it.x + ox, px, f.upem);
    const int peny = to_px((long long)oy - it.y, px, f.upem);    // ベースラインの画素位置
    if (gw > 0 && gh > 0) {
      std::vector<unsigned char> bm((size_t)gw * gh, 0);
      if (g.has_bit(it.cp)) {
        // 持っている画像を目的の大きさに双線形で伸ばす（学習時と同じ拡縮の作法）
        const BitGlyph& bg = g.bits.find(it.cp)->second;
        for (int y = 0; y < gh; ++y)
          for (int x = 0; x < gw; ++x) {
            const float fx = bg.w <= 1 ? 0.f : ((float)x + 0.5f) * bg.w / gw - 0.5f;
            const float fy = bg.h <= 1 ? 0.f : ((float)y + 0.5f) * bg.h / gh - 0.5f;
            const int sx0 = std::max(0, std::min(bg.w - 1, (int)std::floor(fx)));
            const int sy0 = std::max(0, std::min(bg.h - 1, (int)std::floor(fy)));
            const int sx1 = std::min(bg.w - 1, sx0 + 1), sy1 = std::min(bg.h - 1, sy0 + 1);
            const float tx = fx - (float)sx0, ty = fy - (float)sy0;
            const float p00 = bg.pix[(size_t)sy0 * bg.w + sx0];
            const float p10 = bg.pix[(size_t)sy0 * bg.w + sx1];
            const float p01 = bg.pix[(size_t)sy1 * bg.w + sx0];
            const float p11 = bg.pix[(size_t)sy1 * bg.w + sx1];
            const float top = p00 + (p10 - p00) * tx, bot = p01 + (p11 - p01) * tx;
            const float v = top + (bot - top) * ty;
            bm[(size_t)y * gw + x] = (unsigned char)std::max(0.f, std::min(255.f, v));
          }
      } else {
        stbtt_MakeCodepointBitmap(g.info, bm.data(), gw, gh, gw, sc, sc, it.cp);
      }
      for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x) {
          const int dx = penx + gx0 + x, dy = peny + gy0 + y;
          if (dx < 0 || dy < 0 || dx >= R.w || dy >= R.h) continue;
          // 被覆率で紙と字を混ぜる（ink=0, paper=255 なら今までと同じ 255 - 被覆率）
          const int a = bm[(size_t)y * gw + x];
          const int v = st.paper + (st.ink - st.paper) * a / 255;
          unsigned char& dst = R.gray[(size_t)dy * R.w + dx];
          if (v < dst) dst = (unsigned char)v;
        }
    }
    // 正解枠はフォント単位の bbox から作る（ラスタの端の薄い画素に左右されないため）。
    // bbox は絶対座標なので it.x / it.y を足さない（足すと二重加算）
    const int bx0 = to_px((long long)it.x0 + ox, px, f.upem);
    const int bx1 = to_px((long long)it.x1 + ox, px, f.upem);
    const int by0 = to_px((long long)oy - it.y1, px, f.upem);
    const int by1 = to_px((long long)oy - it.y0, px, f.upem);
    R.cls.push_back(it.cls);
    R.box.push_back(bx0); R.box.push_back(by0);
    R.box.push_back(std::max(bx1, bx0 + 1)); R.box.push_back(std::max(by1, by0 + 1));
  }
  // 3x3 の平均で 1 回ぼかす（撮影のぼけと JPEG の甘さの代わり。枠は変わらない）
  if (st.blur) {
    std::vector<unsigned char> src = R.gray;
    for (int y = 0; y < R.h; ++y)
      for (int x = 0; x < R.w; ++x) {
        int sum = 0, n = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || xx < 0 || yy >= R.h || xx >= R.w) continue;
            sum += src[(size_t)yy * R.w + xx];
            ++n;
          }
        R.gray[(size_t)y * R.w + x] = (unsigned char)(sum / n);
      }
  }
  return R;
}

// 式木から描く（今までの入口）。見た目の木を先に作って render_p に渡すだけ
inline Rendered render(const Font& f, const Font* fi, const ex::E& e, int px,
                       const Style& st = Style()) {
  return render_p(f, fi, present(e, false, st), px, st);
}

// 小学校の計算をテキストから描く（÷ や帯分数が畳まれないように、木を経由しない）
inline Rendered render_arith(const Font& f, const Font* fi, const std::string& src, int px,
                             const Style& st = Style()) {
  return render_p(f, fi, present_arith(src, st), px, st);
}

}  // namespace ts
