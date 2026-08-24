// 学習用の式を乱数で作る。
//
// 乱数は `pure/rng.hpp` の splitmix64（このプロジェクトで唯一の乱数）。**同じ種なら C++ と
// Python が同じ式列を出す**ので、どちらでデータセットを作っても同じものになる。
// これは「両言語で学習も推論も評価もできる」を守るための土台で、`tools/parity/dataset.py`
// が実際に突き合わせる。
//
// 何を作るか（最初のゴールの範囲に合わせる）:
//   * 数式（四則・分数・累乗・根号）
//   * 一次方程式・二次方程式（解ける形と、解が無理数になる形の両方）
// 認識の学習に効くのは「見た目の多様性」なので、括弧・入れ子・上付き・分数を混ぜる。
#pragma once
#include "expr.hpp"
#include "rng.hpp"
#include <string>

namespace gx {

inline long long pick(Rng& r, long long lo, long long hi) {
  return lo + (long long)r.below((uint64_t)(hi - lo + 1));
}

// 変数に使う文字。**クラス表にある文字を全部出す**（x, y, t だけだと a・b・p が学習データに
// ほとんど出ず、実写で a が x、b と p が t に化けた）。x を少し厚くするのは教科書がそうだから。
// 26 個ちょうどにしてある（`r.below(26)` の 1 回で引くため。数を変えると式列が変わる）。
inline const char* const kVars[26] = {"x", "x", "x", "x", "y", "y", "y", "t", "t",
                                      "a", "a", "b", "b", "c", "c", "p", "p",
                                      "q", "r", "s", "n", "e", "g", "i", "o", "l"};

// ---------------------------------------------------------------- 小学校の計算
//
// 教科書の書き方（× ÷ を字で書く、小数点、帯分数、中括弧）で 1 行を作る。
// 返すのは **ts::present_arith が読める記法**（frac / mixed）で、同じ文字列を ex::parse に
// 通せば値も出る。**乱数は 1 つずつローカルに受ける**（1 つの式で 2 回呼ぶと C++ の評価順が
// 不定で、Python 側と食い違う。この落とし穴は前に踏んで RESUME に書いてある）。
inline std::string dec(Rng& r, long long lo, long long hi, int digits) {
  const long long ip = pick(r, lo, hi);
  if (digits <= 0) return std::to_string(ip);
  const long long fp = pick(r, 1, digits == 1 ? 9 : 99);
  return std::to_string(ip) + "." + std::to_string(fp);
}

inline std::string arith(Rng& r) {
  const uint64_t kind = r.below(100);
  if (kind < 25) {                                   // 小数の四則
    const std::string a = dec(r, 1, 9, 1);
    const std::string b = dec(r, 1, 9, 1);
    const std::string c = dec(r, 1, 20, 1);
    const std::string d = dec(r, 1, 9, 1);
    return a + " \xc3\x97 (" + b + " - 0." + std::to_string(pick(r, 1, 9)) + ") + " + c +
           " \xc3\xb7 " + d;
  }
  if (kind < 45) {                                   // 中括弧の入った整数の計算
    const long long a = pick(r, 2, 12), b = pick(r, 2, 9), c = pick(r, 10, 40),
                    d = pick(r, 1, 9), e = pick(r, 2, 9);
    return "{" + std::to_string(a) + " \xc3\x97 " + std::to_string(b) + " - (" +
           std::to_string(c) + " - " + std::to_string(d) + ")} \xc3\x97 " + std::to_string(e);
  }
  if (kind < 70) {                                   // 分数と帯分数
    const long long w = pick(r, 1, 5), an = pick(r, 1, 7), ad = pick(r, 2, 9);
    const long long bn = pick(r, 1, 7), bd = pick(r, 2, 9), k = pick(r, 2, 9);
    return "mixed(" + std::to_string(w) + "," + std::to_string(an) + "," + std::to_string(ad) +
           ") \xc3\x97 frac(" + std::to_string(bn) + "," + std::to_string(bd) + ") - frac(" +
           std::to_string(bn) + "," + std::to_string(k) + ")";
  }
  if (kind < 85) {                                   // 分数どうしの割り算
    const long long an = pick(r, 1, 9), ad = pick(r, 2, 9), bn = pick(r, 1, 9),
                    bd = pick(r, 2, 9);
    return "frac(" + std::to_string(an) + "," + std::to_string(ad) + ") \xc3\xb7 frac(" +
           std::to_string(bn) + "," + std::to_string(bd) + ")";
  }
  if (kind < 92) {                                   // 大きい数の掛け算・引き算（103 × 12 - 36）
    const long long a = pick(r, 11, 199), b = pick(r, 2, 24), c = pick(r, 2, 99);
    return std::to_string(a) + " \xc3\x97 " + std::to_string(b) + " - " + std::to_string(c);
  }
  // **文字と × ÷ を同じ絵に出す形。** これが無いと、検出器は「文字が出る絵」と
  // 「× が出る絵」を別々に見ることになり、x と × を見分ける必要がない
  // （実測: 実写で × が x と読まれた。出荷中のモデルは × のクラスすら持っていない）。
  // **l は使わない**（縦棒だけの `1` と見分けられないので、計算問題の文字からは外す。
  // 下の pl::fix_ones が「計算問題で他に文字が無ければ l は 1」と直せるようにするため）
  std::string v = kVars[r.below(26)];
  if (v == "l") v = "x";
  const long long a = pick(r, 2, 12), b = pick(r, 2, 9);
  const uint64_t f = r.below(3);
  if (f == 0) return std::to_string(a) + " \xc3\x97 " + v + " + " + std::to_string(b);
  if (f == 1) return v + " \xc3\x97 " + std::to_string(a) + " - " + std::to_string(b);
  return std::to_string(a) + " \xc3\x97 " + v + " \xc3\xb7 " + std::to_string(b);
}

// 値（数・変数・分数）
inline std::string atom(Rng& r, const std::string& var, int depth) {
  const uint64_t k = r.below(100);
  if (k < 45) return std::to_string(pick(r, -9, 99));
  if (k < 60) {                                   // ここも評価順を固定する（同じ理由。下の term のコメント参照）
    const long long nu = pick(r, 1, 9);
    const long long de = pick(r, 2, 9);
    return std::to_string(nu) + "/" + std::to_string(de);
  }
  if (k < 90) return var;
  if (depth > 0) return "sqrt(" + std::to_string(pick(r, 2, 30)) + ")";
  return std::to_string(pick(r, 2, 12));
}

// 式（深さで打ち切る）
inline std::string term(Rng& r, const std::string& var, int depth) {
  if (depth <= 0) return atom(r, var, depth);
  const uint64_t k = r.below(100);
  // **評価順を明示的に固定する。** `term(...) + " + " + term(...)` と書くと 2 つの呼び出しの
  // 順序は C++ の規格上不定で、実測では MSVC が右から評価した。乱数を消費する関数を式の中で
  // 2 回呼ぶと、コンパイラによって式列が変わる ＝ 「同じ種なら同じデータセット」が崩れる。
  // Python 側との不一致で気付いた（パリティテストが無ければ気付けなかった類の壊れ方）。
  if (k < 22) return atom(r, var, depth);
  if (k < 38) { const std::string a = term(r, var, depth - 1); const std::string b = term(r, var, depth - 1); return a + " + " + b; }
  if (k < 52) { const std::string a = term(r, var, depth - 1); const std::string b = term(r, var, depth - 1); return a + " - " + b; }
  if (k < 64) { const std::string a = term(r, var, depth - 1); const std::string b = term(r, var, depth - 1); return "(" + a + ")(" + b + ")"; }
  if (k < 76) { const std::string a = term(r, var, depth - 1); const std::string b = term(r, var, depth - 1); return "(" + a + ")/(" + b + ")"; }
  if (k < 86) { const std::string a = term(r, var, depth - 1); const long long e = pick(r, 2, 3); return "(" + a + ")^" + std::to_string(e); }
  if (k < 94) return "sqrt(" + term(r, var, depth - 1) + ")";
  { const long long c = pick(r, 2, 9); return std::to_string(c) + var; }
}

// 1 件。方程式にするかどうかも乱数で決める
inline std::string one(Rng& r) {
  // **クラス表にある文字を全部出す。** x, y, t だけで作っていたら、a・b・p などは
  // 学習データに 1 度も出ず、実写で `a` が `x`、`b` と `p` が `t` に化けた（実測）。
  // 重みは実物に近づける（x と y が多く、a・b がそれに次ぐ）。
  const std::string var = kVars[r.below(26)];
  const uint64_t kind = r.below(100);
  if (kind < 35) {                                  // 式だけ（計算問題）
    return term(r, var, 2);
  }
  if (kind < 65) {                                  // 一次方程式
    const long long a = pick(r, 1, 9), b = pick(r, -9, 9), c = pick(r, -9, 9);
    const bool frac = r.below(100) < 30;
    const long long den = pick(r, 2, 5);            // frac でなくても引く（乱数列を揃えるため）
    // 符号は "+ -6" ではなく "- 6" と書く。乱数の呼び方は変えない（式列が変わってしまう）
    std::string s = (a == 1 ? std::string() : std::to_string(a)) + var;
    if (frac) s += "/" + std::to_string(den);
    if (b) s += (b > 0 ? " + " : " - ") + std::to_string(b > 0 ? b : -b);
    return s + " = " + std::to_string(c);
  }
  if (kind < 90) {                                  // 二次方程式（根から作る＝因数分解できる）
    const long long a = r.below(100) < 70 ? 1 : pick(r, 2, 3);
    const long long r1 = pick(r, -6, 6), r2 = pick(r, -6, 6);
    const long long b = -a * (r1 + r2), c = a * r1 * r2;
    std::string s = (a == 1 ? "" : std::to_string(a)) + var + "^2";
    if (b) s += (b > 0 ? " + " : " - ") +
                (b == 1 || b == -1 ? std::string() : std::to_string(b > 0 ? b : -b)) + var;
    if (c) s += (c > 0 ? " + " : " - ") + std::to_string(c > 0 ? c : -c);
    return s + " = 0";
  }
  // 二次方程式（無理数解や実数解なしも出る）
  const long long a = pick(r, 1, 3), b = pick(r, -9, 9), c = pick(r, -9, 9);
  std::string s = (a == 1 ? "" : std::to_string(a)) + var + "^2";
  if (b) s += (b > 0 ? " + " : " - ") +
              (b == 1 || b == -1 ? std::string() : std::to_string(b > 0 ? b : -b)) + var;
  if (c) s += (c > 0 ? " + " : " - ") + std::to_string(c > 0 ? c : -c);
  return s + " = 0";
}

}  // namespace gx
