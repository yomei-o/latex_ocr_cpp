// 数式の組版 — 式木を「画像 + 記号ごとの正解枠」にする。
//
// これは 2 か所で効く 1 つの部品である:
//
//   * **学習データの生成器**。印刷数式の認識器を学習させるのに、データセットを探す必要がない。
//     組版した副産物として**記号ごとの正解枠が厳密に分かる**（人手のアノテーションより正確）。
//   * **手順表示の描画**。solve が出した各段の式を、同じ組版で絵にできる。
//
// 設計判断とその理由:
//
//   * **意味の木（expr）と見た目の木（Pres）を分ける。** `a/b` は意味では `a * b^-1` だが、
//     見た目は分数（横線）である。認識側が読み取るのは見た目の木で、それを意味の木に変換する。
//     この 2 段を混ぜると、レイアウト解析が「意味を推測する」仕事まで背負ってしまう。
//   * **レイアウトはフォント単位の整数で計算する。** 画素に落とすのは最後だけ。こうすると
//     C++ と Python が**同じ整数**を出せる（両方 TTF の hmtx/glyf をそのまま読む）。
//     途中で float を使うと、丸めの違いで枠が 1px ずれ、パリティが「だいたい一致」になる。
//   * **寸法の定数は em の千分率で持つ**（upem が 1000 でも 2048 でも同じ式が使える）。
#pragma once
#include "expr.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// stb は .cpp 側で実装を定義する（ヘッダから include すると実装が二重になる）
struct stbtt_fontinfo;
extern "C" int stbtt_InitFont(stbtt_fontinfo* info, const unsigned char* data, int offset);
extern "C" int stbtt_GetFontOffsetForIndex(const unsigned char* data, int index);
extern "C" void stbtt_GetFontVMetrics(const stbtt_fontinfo* info, int* ascent, int* descent,
                                      int* lineGap);
extern "C" void stbtt_GetCodepointHMetrics(const stbtt_fontinfo* info, int codepoint,
                                           int* advanceWidth, int* leftSideBearing);
extern "C" int stbtt_GetCodepointBox(const stbtt_fontinfo* info, int codepoint, int* x0, int* y0,
                                     int* x1, int* y1);
extern "C" void stbtt_GetCodepointBitmapBox(const stbtt_fontinfo* info, int codepoint,
                                            float scale_x, float scale_y, int* ix0, int* iy0,
                                            int* ix1, int* iy1);
extern "C" void stbtt_MakeCodepointBitmap(const stbtt_fontinfo* info, unsigned char* output,
                                          int out_w, int out_h, int out_stride, float scale_x,
                                          float scale_y, int codepoint);
extern "C" float stbtt_ScaleForPixelHeight(const stbtt_fontinfo* info, float pixels);

namespace ts {

// em の千分率で持つ寸法（値の出どころは TeX の慣習に近い比率。実測で詰める余地はある）
enum : int {
  K_SUP_SHIFT = 450,      // 上付きをベースラインから持ち上げる量
  K_SUP_NUM = 7,          // 上付きの縮小率 7/10
  K_SUP_DEN = 10,
  K_AXIS = 280,           // 分数の横線の高さ（ベースラインから）
  K_BAR = 50,             // 横線の太さ
  K_FRAC_GAP = 120,       // 横線と分子・分母の隙間
  K_FRAC_PAD = 80,        // 分数の左右の余白
  K_OP_SPACE = 180,       // + - = の左右の空き
  K_SQRT_PAD = 60,        // 根号の中身の左右の余白
  K_MARGIN = 250,         // 画像の余白
};

// ---------------------------------------------------------------- 見た目の木

enum class PK { Glyph, Row, Sup, Frac, Sqrt, Paren };

struct PNode;
using P = std::shared_ptr<PNode>;

struct PNode {
  PK k = PK::Glyph;
  int cp = 0;                 // Glyph の文字（Unicode）
  std::string cls;            // 認識のクラス名（"0".."9", "x", "+", "sqrt", "frac" …）
  bool ital = false;          // イタリックの書体で描く字か（教科書は変数だけイタリック）
  std::vector<P> kids;
};

inline P pg(int cp, const std::string& cls, bool ital = false) {
  auto n = std::make_shared<PNode>();
  n->k = PK::Glyph; n->cp = cp; n->cls = cls; n->ital = ital;
  return n;
}
inline P pn(PK k, std::vector<P> kids) {
  auto n = std::make_shared<PNode>();
  n->k = k; n->kids = std::move(kids);
  return n;
}

// 描き方の指定。**実写の教科書に寄せるためのつまみ**で、既定は今までと同じ絵。
//
//   * italic_vars: 変数をイタリックで描く。教科書は必ずそう組む（数字は立体、文字は斜体）。
//   * minus_cp: マイナスの字。教科書は U+2212（長い横棒）で、ASCII の '-' より明らかに長い。
//     実測: 本物の写真で U+2212 のマイナスは conf 0.01〜0.05 しか出なかった（学習データに
//     '-' しか無いため）。クラス名は "-" のままなので、クラス表は増えない。
//   * ink / paper: 字と紙の明るさ。写真は**真っ黒と真っ白ではない**（実測: 教科書の写真は
//     紙 225 前後、字 60 前後）。既定は 0 / 255 なので、今までの絵と同じものが出る。
//   * blur: 1 なら 3x3 の平均で 1 回ぼかす（撮影のぼけと JPEG の甘さの代わり）。
struct Style {
  bool italic_vars = false;
  int minus_cp = '-';
  int ink = 0;
  int paper = 255;
  int blur = 0;
  // **数字の 1 を「縦棒だけ」の字で描く。** 実写の教科書の 1 は旗も台も無い縦棒で
  // （実測: `103 × 12 - 36` の 1 が小文字 l と読まれた）、手元の書体はどれも旗つき。
  // 学習データにこの字形が 1 つも無いと、モデルは縦棒を l と読むしかない。
  // ラベルは "1" のまま、**字だけ l のグリフを借りる**（Century Gothic の 1 と同じ形）。
  bool plain_one = false;
};

inline void push_digits(std::vector<P>& out, long long v, const Style& st = Style()) {
  if (v < 0) { out.push_back(pg(st.minus_cp, "-")); v = -v; }
  std::string s = std::to_string(v);
  for (char c : s)
    out.push_back(pg(st.plain_one && c == '1' ? 'l' : c, std::string(1, c)));
}

// 意味の木 -> 見た目の木。ここが「a/b は分数として描く」を決める場所。
inline P present(const ex::E& e, bool paren = false, const Style& st = Style());

inline P present_row(const std::vector<P>& items) { return pn(PK::Row, items); }

// 積を「分子の因子」と「分母の因子」に分ける（指数が負のものが分母）
inline void split_frac(const ex::E& e, std::vector<ex::E>& up, std::vector<ex::E>& down) {
  using namespace ex;
  std::vector<E> fs;
  if (e->k == Kind::Mul) fs = e->kids; else fs.push_back(e);
  for (const E& f : fs) {
    if (f->k == Kind::Pow && is_num(f->kids[1]) && f->kids[1]->num.neg()) {
      const Rat p = f->kids[1]->num;
      down.push_back(p.n == -1 && p.d == 1 ? f->kids[0]
                                           : raw(Kind::Pow, {f->kids[0], num(-p)}));
    } else if (is_num(f) && !f->num.is_int()) {
      up.push_back(num(ex::Rat(f->num.n)));
      down.push_back(num(ex::Rat(f->num.d)));
    } else {
      up.push_back(f);
    }
  }
}

// **その因子は数字から描き始まるか**（掛け算の記号を省いてよいかの判定に使う）。
// 数のあとに数が並ぶと、人にも機械にも桁として読める（実測: `8 * 6^(3/2)` を並べて描くと
// `86^(3/2)` と読み戻され、往復不変が壊れた）。根号や分数から始まるものは衝突しない。
inline bool starts_with_digit(const ex::E& f) {
  using namespace ex;
  if (is_num(f)) return f->num.is_int();
  if (f->k == Kind::Pow) {
    if (is_num(f->kids[1]) && (f->kids[1]->num == Rat(1, 2) || f->kids[1]->num.neg()))
      return false;                                  // 根号・分数で描かれる
    return starts_with_digit(f->kids[0]);
  }
  if (f->k == Kind::Mul && !f->kids.empty()) return starts_with_digit(f->kids[0]);
  return false;
}

inline P present(const ex::E& e, bool paren, const Style& st) {
  using namespace ex;
  std::vector<P> row;
  switch (e->k) {
    case Kind::Num: {
      if (e->num.is_int()) { push_digits(row, e->num.n, st); break; }
      // 分数は横線で描く（意味の木では有理数 1 個でも、見た目は 2 段）
      std::vector<P> nu, de;
      push_digits(nu, e->num.n, st);
      push_digits(de, e->num.d, st);
      return pn(PK::Frac, {present_row(nu), present_row(de)});
    }
    case Kind::Sym:
      // 変数だけイタリック（教科書の組み方）
      for (char c : e->name) row.push_back(pg(c, std::string(1, c), st.italic_vars));
      break;
    case Kind::Add: {
      const std::vector<E> ts = disp_terms(e);
      for (size_t i = 0; i < ts.size(); ++i) {
        Rat c; E rest;
        split_coeff(ts[i], c, rest);
        const bool minus = c.neg();
        if (i == 0) { if (minus) row.push_back(pg(st.minus_cp, "-")); }
        else row.push_back(pg(minus ? st.minus_cp : '+', minus ? "-" : "+"));
        Rat ac = minus ? -c : c;
        E body = ac.is_one() && !is_num(rest) ? rest
                 : (is_num(rest) && rest->num.is_one() ? num(ac) : mul_n({num(ac), rest}));
        row.push_back(present(body, body->k == Kind::Add, st));
      }
      break;
    }
    case Kind::Mul: {
      std::vector<E> up, down;
      split_frac(e, up, down);
      if (!down.empty()) {
        // 分子の 1 は書かない（1/2 x は "1x/2" ではなく "x/2"）。ただし分子が 1 だけなら残す
        if (up.size() > 1) {
          std::vector<E> keep;
          for (const E& u : up)
            if (!(is_num(u) && u->num.is_one())) keep.push_back(u);
          if (!keep.empty()) up = keep;
        }
        const E nu = up.empty() ? num(Rat(1)) : (up.size() == 1 ? up[0] : raw(Kind::Mul, up));
        const E de = down.size() == 1 ? down[0] : raw(Kind::Mul, down);
        return pn(PK::Frac, {present(nu, false, st), present(de, false, st)});
      }
      // 掛け算は記号を書かずに並べる（印刷数式の慣習。2x, 3(x+1)）。
      // **ただし数のあとに数が来るときは × を書く**（並べると桁として読める）
      for (size_t i = 0; i < e->kids.size(); ++i) {
        const E& f = e->kids[i];
        if (i && is_num(e->kids[i - 1]) && starts_with_digit(f))
          row.push_back(pg(0x00D7, "times"));
        row.push_back(present(f, f->k == Kind::Add, st));
      }
      break;
    }
    case Kind::Pow: {
      const E& b = e->kids[0];
      const E& p = e->kids[1];
      if (is_num(p) && p->num == Rat(1, 2)) return pn(PK::Sqrt, {present(b, false, st)});
      // **負の指数は分数で描く**（人は y^-1 ではなく 1/y と書く）。上付きで描くと、
      // 解析側が指数の "-" を二項の引き算と取り違える（実測: 1/y が y - 1 になった）。
      if (is_num(p) && p->num.neg()) {
        const E inv = p->num.n == -1 && p->num.d == 1 ? b : raw(Kind::Pow, {b, num(-p->num)});
        return pn(PK::Frac, {present(num(Rat(1)), false, st), present(inv, false, st)});
      }
      return pn(PK::Sup, {present(b, b->k == Kind::Add || b->k == Kind::Mul, st),
                          present(p, false, st)});
    }
    case Kind::Fn: {
      // **引数は全部描く**（"," 区切り）。1 つしか描かないと sum(k, 1, n, 中身) が
      // "sum(k)" になる。Σ を大きな記号と上下の添字で描くのは認識側の仕事（クラスに
      // Σ が無いので、勝手に字を増やすと学習データと食い違う）。いまは書いたとおりに描く。
      for (char c : e->name) row.push_back(pg(c, std::string(1, c)));
      if (e->kids.empty()) break;                    // 定数（pi）は括弧を付けない
      std::vector<P> args;
      for (size_t i = 0; i < e->kids.size(); ++i) {
        if (i) args.push_back(pg(',', ","));
        args.push_back(present(e->kids[i], false, st));
      }
      if (args.empty()) args.push_back(present(num(Rat(0)), false, st));
      row.push_back(pn(PK::Paren, {present_row(args)}));
      break;
    }
    case Kind::Rel:
      // 演算子は name の 1 文字ずつを字として置く。"=" のときは以前と同じ絵になる。
      // **"<=" は暫定で '<' '=' の 2 字**（本物の ≤ の字と、そのクラス追加は認識側の仕事。
      // いまの検出器のクラス表に ≤ が無いので、勝手に字を増やすと学習データと食い違う）。
      row.push_back(present(e->kids[0], false, st));
      for (char c : e->name) row.push_back(pg(c == '-' ? st.minus_cp : c, std::string(1, c)));
      row.push_back(present(e->kids[1], false, st));
      break;
    case Kind::Sys:
      // 連立は暫定で 1 行に "," 区切り（本来は中括弧つきの複数行。行の切り分けと
      // '{' のクラスが要るので、認識側と一緒に作る）
      for (size_t i = 0; i < e->kids.size(); ++i) {
        if (i) row.push_back(pg(',', ","));
        row.push_back(present(e->kids[i], false, st));
      }
      break;
  }
  P r = present_row(row);
  return paren ? pn(PK::Paren, {r}) : r;
}

// ---------------------------------------------------------------- 小学校の計算を描く
//
// **CAS の木からは描けない。** `0.96 ÷ 1.2` は正規形にすると `4/5` に畳まれて ÷ が消えるし、
// `2 5/8` も `21/8` になる。書かれたとおりの絵が欲しいので、**テキストから直に見た目の木を作る**。
// 値のほうは同じテキストを ex::parse に通せば得られる（frac / mixed も読める）ので、
// 「絵」と「値」の出どころは 1 つのテキストに保たれる。
//
// 記法: 数（小数点あり）・`+ - × ÷ ( ) { }`、`frac(a,b)` は縦の分数、`mixed(w,a,b)` は帯分数。
inline P present_arith(const std::string& s, size_t& i, const Style& st, char stop = 0);

inline P arith_atom(const std::string& s, size_t& i, const Style& st) {
  std::vector<P> row;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  if (i >= s.size()) return pn(PK::Row, row);
  // frac(a,b) と mixed(w,a,b)
  if (s.compare(i, 5, "frac(") == 0) {
    i += 5;
    P a = present_arith(s, i, st, ',');
    if (i < s.size() && s[i] == ',') ++i;
    P b = present_arith(s, i, st, ')');
    if (i < s.size() && s[i] == ')') ++i;
    return pn(PK::Frac, {a, b});
  }
  if (s.compare(i, 6, "mixed(") == 0) {
    i += 6;
    P w = present_arith(s, i, st, ',');
    if (i < s.size() && s[i] == ',') ++i;
    P a = present_arith(s, i, st, ',');
    if (i < s.size() && s[i] == ',') ++i;
    P b = present_arith(s, i, st, ')');
    if (i < s.size() && s[i] == ')') ++i;
    return pn(PK::Row, {w, pn(PK::Frac, {a, b})});     // 帯分数は整数と分数を並べるだけ
  }
  if (s[i] == '(') {
    ++i;
    P in = present_arith(s, i, st, ')');
    if (i < s.size() && s[i] == ')') ++i;
    return pn(PK::Paren, {in});
  }
  if (s[i] == '{') {                                   // 中括弧は伸ばさない（教科書もそう）
    ++i;
    P in = present_arith(s, i, st, '}');
    if (i < s.size() && s[i] == '}') ++i;
    row.push_back(pg('{', "brace_l"));
    row.push_back(in);
    row.push_back(pg('}', "brace_r"));
    return pn(PK::Row, row);
  }
  if (isdigit((unsigned char)s[i]) || s[i] == '.') {
    while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.')) {
      const char c = s[i++];
      row.push_back(c == '.' ? pg('.', "dot") : pg(c, std::string(1, c)));
    }
    return pn(PK::Row, row);
  }
  if (isalpha((unsigned char)s[i])) {                  // 文字（面積の cm などは入れない）
    row.push_back(pg(s[i], std::string(1, s[i]), st.italic_vars));
    ++i;
    return pn(PK::Row, row);
  }
  ++i;                                                 // 読めない字は捨てる
  return pn(PK::Row, row);
}

inline P present_arith(const std::string& s, size_t& i, const Style& st, char stop) {
  std::vector<P> row;
  for (;;) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i >= s.size()) break;
    if (stop && s[i] == stop) break;
    if (s[i] == ')' || s[i] == '}' || s[i] == ',') break;
    if (s[i] == '+') { row.push_back(pg('+', "+")); ++i; continue; }
    if (s[i] == '-') { row.push_back(pg(st.minus_cp, "-")); ++i; continue; }
    if (s[i] == '=') { row.push_back(pg('=', "=")); ++i; continue; }
    if (s.compare(i, 2, "\xc3\x97") == 0) { row.push_back(pg(0x00D7, "times")); i += 2; continue; }
    if (s.compare(i, 2, "\xc3\xb7") == 0) { row.push_back(pg(0x00F7, "div")); i += 2; continue; }
    row.push_back(arith_atom(s, i, st));
  }
  return pn(PK::Row, row);
}

inline P present_arith(const std::string& s, const Style& st = Style()) {
  size_t i = 0;
  return present_arith(s, i, st, 0);
}

// ---------------------------------------------------------------- フォント

// **画像で持つ字。** 教科書体のように再配布できない書体でも、必要な字だけ被覆率の画像に
// して持てば学習データを作れる（字の形だけを使う。書体ファイルは配らない）。
// 小学校のページに要るのは数字・四則・括弧なので、そこだけ差し替える使い方を想定している。
struct BitGlyph {
  int adv = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // 書き出したときのフォント単位
  int w = 0, h = 0;                               // 画像の大きさ（bbox にぴったり）
  std::vector<unsigned char> pix;                 // 被覆率 0..255（w*h）
};

struct Font {
  std::vector<unsigned char> data;
  stbtt_fontinfo* info = nullptr;      // .cpp 側で確保する（stb の型が不完全なので）
  int upem = 1000, ascent = 800, descent = 200;
  // 画像で差し替える字（符号位置 -> 字）。bit_upem は書き出し元の upem
  int bit_upem = 0;
  std::map<int, BitGlyph> bits;
  bool has_bit(int cp) const { return bit_upem > 0 && bits.find(cp) != bits.end(); }
  int advance(int cp) const;
  void bbox(int cp, int* x0, int* y0, int* x1, int* y1) const;
};

// em の千分率 -> フォント単位
inline int emk(const Font& f, int k) { return (int)((long long)f.upem * k / 1000); }

// ---------------------------------------------------------------- レイアウト

// 1 つの箱。すべてフォント単位。asc/desc はベースラインからの上下。
struct Box {
  int w = 0, asc = 0, desc = 0;
};

// 置かれた記号（フォント単位。画素にするのは最後）
struct Item {
  std::string cls;
  int cp = 0;                 // 0 なら「線」（分数の横線・根号の横線）
  int x = 0, y = 0;           // ベースライン原点（線のときは矩形の左下）
  int scale_num = 1, scale_den = 1;   // 縮小率（上付きなど）
  int w = 0, h = 0;           // 線のときの矩形の大きさ
  bool ital = false;          // イタリックの書体で描く
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0; // 正解枠（フォント単位、y は上向き正）
};

struct Layout {
  Box box;
  std::vector<Item> items;
};

// scale は「この部分木を何倍で描くか」を分数で持つ（整数演算のまま縮小するため）
Layout lay(const Font& f, const Font* fi, const P& p, int sn, int sd);

inline int mulr(int v, int sn, int sd) { return (int)((long long)v * sn / sd); }

inline void shift(Layout& L, int dx, int dy) {
  for (Item& it : L.items) {
    it.x += dx; it.y += dy;
    it.x0 += dx; it.x1 += dx; it.y0 += dy; it.y1 += dy;
  }
}

// 別の upem を持つ書体を混ぜても崩れないように、字幅を「ローマン体のフォント単位」に直す
inline int conv_upem(int v, int from_upem, int to_upem) {
  return (int)((long long)v * to_upem / from_upem);
}

inline Layout lay(const Font& f, const Font* fi, const P& p, int sn, int sd) {
  Layout out;
  switch (p->k) {
    case PK::Glyph: {
      // イタリックの字は第 2 書体から寸法を取り、**ローマン体のフォント単位に直す**
      // （upem が違う書体を混ぜると、直さないと字だけ拡大縮小されてしまう）
      const bool ital = p->ital && fi != nullptr;
      const Font& g = ital ? *fi : f;
      int adv = g.advance(p->cp);
      int x0, y0, x1, y1;
      g.bbox(p->cp, &x0, &y0, &x1, &y1);
      if (ital && g.upem != f.upem) {
        adv = conv_upem(adv, g.upem, f.upem);
        x0 = conv_upem(x0, g.upem, f.upem);
        y0 = conv_upem(y0, g.upem, f.upem);
        x1 = conv_upem(x1, g.upem, f.upem);
        y1 = conv_upem(y1, g.upem, f.upem);
      }
      adv = mulr(adv, sn, sd);
      Item it;
      it.cls = p->cls;
      it.cp = p->cp;
      it.ital = ital;
      it.x = 0; it.y = 0;
      it.scale_num = sn; it.scale_den = sd;
      it.x0 = mulr(x0, sn, sd); it.y0 = mulr(y0, sn, sd);
      it.x1 = mulr(x1, sn, sd); it.y1 = mulr(y1, sn, sd);
      out.items.push_back(it);
      out.box.w = adv;
      out.box.asc = it.y1 > 0 ? it.y1 : 0;
      out.box.desc = it.y0 < 0 ? -it.y0 : 0;
      return out;
    }
    case PK::Row: {
      int x = 0, asc = 0, desc = 0;
      for (const P& c : p->kids) {
        Layout L = lay(f, fi, c, sn, sd);
        const bool op = c->k == PK::Glyph && (c->cls == "+" || c->cls == "-" || c->cls == "=");
        if (op) x += mulr(emk(f, K_OP_SPACE), sn, sd);
        shift(L, x, 0);
        x += L.box.w;
        if (op) x += mulr(emk(f, K_OP_SPACE), sn, sd);
        asc = asc > L.box.asc ? asc : L.box.asc;
        desc = desc > L.box.desc ? desc : L.box.desc;
        for (Item& it : L.items) out.items.push_back(it);
      }
      out.box.w = x; out.box.asc = asc; out.box.desc = desc;
      return out;
    }
    case PK::Sup: {
      Layout b = lay(f, fi, p->kids[0], sn, sd);
      Layout e = lay(f, fi, p->kids[1], sn * K_SUP_NUM, sd * K_SUP_DEN);
      const int rise = mulr(emk(f, K_SUP_SHIFT), sn, sd);
      shift(e, b.box.w, rise);
      out.items = b.items;
      for (Item& it : e.items) out.items.push_back(it);
      out.box.w = b.box.w + e.box.w;
      out.box.asc = std::max(b.box.asc, rise + e.box.asc);
      out.box.desc = std::max(b.box.desc, e.box.desc - rise > 0 ? e.box.desc - rise : 0);
      return out;
    }
    case PK::Frac: {
      Layout nu = lay(f, fi, p->kids[0], sn, sd);
      Layout de = lay(f, fi, p->kids[1], sn, sd);
      const int pad = mulr(emk(f, K_FRAC_PAD), sn, sd);
      const int bar = mulr(emk(f, K_BAR), sn, sd);
      const int gap = mulr(emk(f, K_FRAC_GAP), sn, sd);
      const int axis = mulr(emk(f, K_AXIS), sn, sd);
      const int inner = std::max(nu.box.w, de.box.w);
      const int w = inner + 2 * pad;
      shift(nu, pad + (inner - nu.box.w) / 2, axis + bar + gap + nu.box.desc);
      shift(de, pad + (inner - de.box.w) / 2, axis - gap - de.box.asc);
      Item line;
      line.cls = "frac";
      line.cp = 0;
      line.x = 0; line.y = axis;
      line.w = w; line.h = bar;
      line.x0 = 0; line.y0 = axis; line.x1 = w; line.y1 = axis + bar;
      out.items.push_back(line);
      for (Item& it : nu.items) out.items.push_back(it);
      for (Item& it : de.items) out.items.push_back(it);
      out.box.w = w;
      out.box.asc = axis + bar + gap + nu.box.desc + nu.box.asc;
      out.box.desc = std::max(0, -(axis - gap - de.box.asc - de.box.desc));
      return out;
    }
    case PK::Sqrt: {
      Layout in = lay(f, fi, p->kids[0], sn, sd);
      const int pad = mulr(emk(f, K_SQRT_PAD), sn, sd);
      const int radv = mulr(f.advance(0x221A), sn, sd);
      const int bar = mulr(emk(f, K_BAR), sn, sd);
      // 根号そのもの（記号）と、中身の上に伸びる横線
      Layout rad = lay(f, fi, pg(0x221A, "sqrt"), sn, sd);
      shift(in, radv + pad, 0);
      const int top = std::max(in.box.asc + pad, rad.box.asc);
      Item line;
      line.cls = "sqrt";
      line.cp = 0;
      line.x = radv; line.y = top;
      line.w = in.box.w + 2 * pad; line.h = bar;
      line.x0 = radv; line.y0 = top; line.x1 = radv + in.box.w + 2 * pad; line.y1 = top + bar;
      out.items = rad.items;
      out.items.push_back(line);
      for (Item& it : in.items) out.items.push_back(it);
      out.box.w = radv + in.box.w + 2 * pad;
      out.box.asc = top + bar;
      out.box.desc = std::max(rad.box.desc, in.box.desc);
      return out;
    }
    case PK::Paren: {
      Layout in = lay(f, fi, p->kids[0], sn, sd);
      Layout l = lay(f, fi, pg('(', "("), sn, sd);
      Layout r = lay(f, fi, pg(')', ")"), sn, sd);
      shift(in, l.box.w, 0);
      shift(r, l.box.w + in.box.w, 0);
      out.items = l.items;
      for (Item& it : in.items) out.items.push_back(it);
      for (Item& it : r.items) out.items.push_back(it);
      out.box.w = l.box.w + in.box.w + r.box.w;
      out.box.asc = std::max(std::max(l.box.asc, r.box.asc), in.box.asc);
      out.box.desc = std::max(std::max(l.box.desc, r.box.desc), in.box.desc);
      return out;
    }
  }
  return out;
}

// ---------------------------------------------------------------- 画素に落とす

// フォント単位 -> 画素。両言語で同じ丸めをする（負の値でも同じになるように書く）
inline int to_px(long long v, int px, int upem) {
  const long long num = v * px * 2;
  const long long den = (long long)upem * 2;
  return (int)(num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den));
}

struct Rendered {
  int w = 0, h = 0;
  std::vector<unsigned char> gray;                 // 白地に黒字（0=黒, 255=白）
  std::vector<std::string> cls;                    // 記号のクラス
  std::vector<int> box;                            // x0,y0,x1,y1（画素、y は下向き正）を 4 個ずつ
};

}  // namespace ts
