// 数式そのもの — 木・厳密有理数・正規形・印字・構文解析。
//
// このファイルが決めることが、後の全部の契約になる:
//
//   * 認識側（写真 -> 記号 -> レイアウト解析）の出力はこの木である。
//   * 解く側（一次・二次方程式、将来の微分積分）はこの木を書き換える。
//   * 手順表示は「名前のついた書き換え」を記録したものである。
//
// 設計判断とその理由:
//
//   * **数は厳密有理数**（int64 の分子・分母、既約、分母は正）。1/3 を 0.333 にしたら
//     手順表示が嘘になる。double は最後の表示でしか使わない。
//   * **Add と Mul は可変長**（入れ子を平らにする）。正規形で同類項をまとめるのに必要で、
//     二項木のままだと (a+b)+c と a+(b+c) が別物になる。
//   * **木は不変で共有する**（shared_ptr<const Node>）。書き換えは新しい木を作る。
//     手順表示は「各段の木」を保持するので、破壊的更新だと過去の段が壊れる。
//   * **正規形は全順序でソートする**。そうすると「同じ式か」が構造の一致で判定でき、
//     x + 1 と 1 + x が同一になる。順序は決定的（種類 -> 名前 -> 子の数 -> 子）。
//   * **簡約は手順に出さない**。手順に出すのは solve が選んだ規則（「両辺から 3 を引く」
//     「解の公式」）だけ。正規化の 1 手 1 手を見せると、人間には読めないものになる。
//     Photomath 系アプリの価値は「読める手順」なので、ここは意図的に分けてある。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <cstring>
#include <string>
#include <vector>

namespace ex {

// ---------------------------------------------------------------- 厳密有理数

struct Rat {
  long long n = 0, d = 1;

  Rat() = default;
  Rat(long long v) : n(v), d(1) {}
  Rat(long long num, long long den) : n(num), d(den) { norm(); }

  void norm() {
    if (d == 0) { n = 0; d = 1; return; }          // 0 除算は 0 に落とす（呼ぶ側で弾く）
    if (d < 0) { n = -n; d = -d; }
    long long g = std::gcd(n < 0 ? -n : n, d);
    if (g > 1) { n /= g; d /= g; }
    if (n == 0) d = 1;
  }
  bool is_int() const { return d == 1; }
  bool is_zero() const { return n == 0; }
  bool is_one() const { return n == 1 && d == 1; }
  bool neg() const { return n < 0; }
  double f() const { return (double)n / (double)d; }

  Rat operator+(const Rat& o) const { return Rat(n * o.d + o.n * d, d * o.d); }
  Rat operator-(const Rat& o) const { return Rat(n * o.d - o.n * d, d * o.d); }
  Rat operator*(const Rat& o) const { return Rat(n * o.n, d * o.d); }
  Rat operator/(const Rat& o) const { return Rat(n * o.d, d * o.n); }
  Rat operator-() const { Rat r; r.n = -n; r.d = d; return r; }
  bool operator==(const Rat& o) const { return n == o.n && d == o.d; }
  bool operator<(const Rat& o) const { return n * o.d < o.n * d; }

  std::string str() const {
    char b[64];
    if (d == 1) snprintf(b, sizeof b, "%lld", n);
    else snprintf(b, sizeof b, "%lld/%lld", n, d);
    return b;
  }
};

// 整数の冪。指数が負や分数のときは呼ばない（呼ぶ側が Pow のまま残す）
inline Rat rpow(const Rat& a, long long e) {
  if (e < 0) return Rat(1) / rpow(a, -e);
  Rat r(1), b = a;
  for (long long i = 0; i < e; ++i) r = r * b;
  return r;
}

// ---------------------------------------------------------------- 木

// Rel は関係式（= < <= > >=）で、**演算子は name に入れる**。Kind を 5 つに分けない理由は、
// cmp / simp / expand / 印字が全部同じ形で書けるから（既定の cmp が name も比べている）。
// Sys は連立（関係式の列）。並びは書かれた順のまま保つ（人が書いた順で手順を見せるため）。
enum class Kind { Num, Sym, Add, Mul, Pow, Fn, Rel, Sys };

struct Node;
using E = std::shared_ptr<const Node>;

struct Node {
  Kind k = Kind::Num;
  Rat num;                    // Num
  std::string name;           // Sym の変数名 / Fn の関数名 / Rel の演算子
  std::vector<E> kids;        // Add/Mul は 2 個以上、Pow と Rel は 2 個、Fn は引数、Sys は関係式
};

inline E mk(Kind k) { auto n = std::make_shared<Node>(); n->k = k; return n; }
inline E num(const Rat& r) { auto n = std::make_shared<Node>(); n->k = Kind::Num; n->num = r; return n; }
inline E num(long long v) { return num(Rat(v)); }
inline E sym(const std::string& s) {
  auto n = std::make_shared<Node>(); n->k = Kind::Sym; n->name = s; return n;
}
inline E raw(Kind k, std::vector<E> kids, const std::string& name = "") {
  auto n = std::make_shared<Node>();
  n->k = k; n->name = name; n->kids = std::move(kids);
  return n;
}

inline bool is_num(const E& e) { return e->k == Kind::Num; }
inline bool is_sym(const E& e) { return e->k == Kind::Sym; }

// ---------------------------------------------------------------- 全順序

// 決定的な順序。正規形のソートに使い、同じ式が構造として一致するようにする。
inline int cmp(const E& a, const E& b) {
  if (a->k != b->k) return (int)a->k < (int)b->k ? -1 : 1;
  switch (a->k) {
    case Kind::Num:
      if (a->num == b->num) return 0;
      return a->num < b->num ? -1 : 1;
    case Kind::Sym:
      return a->name < b->name ? -1 : (a->name == b->name ? 0 : 1);
    default: {
      if (a->name != b->name) return a->name < b->name ? -1 : 1;
      if (a->kids.size() != b->kids.size()) return a->kids.size() < b->kids.size() ? -1 : 1;
      for (size_t i = 0; i < a->kids.size(); ++i) {
        int c = cmp(a->kids[i], b->kids[i]);
        if (c) return c;
      }
      return 0;
    }
  }
}
inline bool equal(const E& a, const E& b) { return cmp(a, b) == 0; }

// ---------------------------------------------------------------- 正規形

E simp(const E& e);

inline void flatten(Kind k, const E& e, std::vector<E>& out) {
  if (e->k == k) { for (const E& c : e->kids) flatten(k, c, out); }
  else out.push_back(e);
}

// 項を「係数 × 残り」に割る（同類項をまとめるため）
inline void split_coeff(const E& t, Rat& c, E& rest) {
  c = Rat(1);
  if (t->k == Kind::Num) { c = t->num; rest = num(Rat(1)); return; }
  if (t->k != Kind::Mul) { rest = t; return; }
  std::vector<E> others;
  for (const E& f : t->kids) {
    if (f->k == Kind::Num) c = c * f->num;
    else others.push_back(f);
  }
  if (others.empty()) rest = num(Rat(1));
  else if (others.size() == 1) rest = others[0];
  else rest = raw(Kind::Mul, others);
}

inline E pow_e(const E& b_in, const E& e_in);

// 掛け算のあふれ検査（根の中身 n^p' * d^(q-p') を作るときに使う）
inline bool mul_safe(long long a, long long b, long long& out) {
  if (a == 0 || b == 0) { out = 0; return true; }
  if (a > 4000000000000000LL / (b > 0 ? b : -b)) return false;
  out = a * b;
  return true;
}

// 「数の 1/q 乗」か（sqrt(2) や 6^(1/3)）。そうなら次数 q と中身を返す
inline bool num_radical(const E& f, long long& q, Rat& inside) {
  if (f->k != Kind::Pow || !is_num(f->kids[0]) || !is_num(f->kids[1])) return false;
  if (f->kids[1]->num.n != 1 || f->kids[1]->num.d <= 1) return false;
  q = f->kids[1]->num.d;
  inside = f->kids[0]->num;
  return true;
}

// 因子を「底 ^ 指数」に割る（同じ底をまとめるため）
inline void split_pow(const E& f, E& base, E& expo) {
  if (f->k == Kind::Pow) { base = f->kids[0]; expo = f->kids[1]; }
  else { base = f; expo = num(Rat(1)); }
}

inline E add_n(std::vector<E> xs) {
  std::vector<E> flat;
  for (E& x : xs) flatten(Kind::Add, simp(x), flat);
  Rat konst(0);
  std::vector<std::pair<E, Rat>> terms;              // 残り -> 係数の合計
  for (const E& t : flat) {
    if (t->k == Kind::Num) { konst = konst + t->num; continue; }
    Rat c; E rest;
    split_coeff(t, c, rest);
    bool merged = false;
    for (auto& pr : terms)
      if (equal(pr.first, rest)) { pr.second = pr.second + c; merged = true; break; }
    if (!merged) terms.emplace_back(rest, c);
  }
  std::vector<E> out;
  for (auto& pr : terms) {
    if (pr.second.is_zero()) continue;
    if (pr.second.is_one()) out.push_back(pr.first);
    else out.push_back(simp(raw(Kind::Mul, {num(pr.second), pr.first})));
  }
  if (!konst.is_zero()) out.push_back(num(konst));
  if (out.empty()) return num(Rat(0));
  if (out.size() == 1) return out[0];
  std::sort(out.begin(), out.end(), [](const E& a, const E& b) { return cmp(a, b) < 0; });
  return raw(Kind::Add, out);
}

inline E mul_n(std::vector<E> xs) {
  std::vector<E> flat;
  for (E& x : xs) flatten(Kind::Mul, simp(x), flat);
  Rat coef(1);
  std::vector<std::pair<E, std::vector<E>>> bases;   // 底 -> 指数たち
  for (const E& f : flat) {
    if (f->k == Kind::Num) {
      coef = coef * f->num;
      if (coef.is_zero()) return num(Rat(0));
      continue;
    }
    E b, p;
    split_pow(f, b, p);
    bool merged = false;
    for (auto& pr : bases)
      if (equal(pr.first, b)) { pr.second.push_back(p); merged = true; break; }
    if (!merged) bases.emplace_back(b, std::vector<E>{p});
  }
  std::vector<E> out;
  for (auto& pr : bases) {
    E e = pr.second.size() == 1 ? pr.second[0] : add_n(pr.second);
    if (is_num(e) && e->num.is_zero()) continue;                 // x^0 = 1
    if (is_num(e) && e->num.is_one()) { out.push_back(pr.first); continue; }
    out.push_back(raw(Kind::Pow, {pr.first, e}));
  }
  // **同じ次数の根はまとめる**（sqrt(2)*sqrt(3) = sqrt(6)）。教科書はそう書くし、
  // まとめないと sqrt(3)/sqrt(2) が有理化されない（1/sqrt(2) は sqrt(2)/2 になるのに、
  // 根どうしの割り算だけ残ってしまう）。
  {
    std::vector<std::pair<long long, Rat>> rads;     // 次数 q -> 中身の積（現れた順）
    std::vector<E> keep;
    bool merged = false;
    for (const E& f : out) {
      long long q = 0;
      Rat in;
      if (num_radical(f, q, in)) {
        bool hit = false;
        for (auto& pr : rads)
          if (pr.first == q) { pr.second = pr.second * in; hit = merged = true; break; }
        if (!hit) rads.emplace_back(q, in);
        continue;
      }
      keep.push_back(f);
    }
    if (merged) {
      std::vector<E> xs;
      if (!coef.is_one()) xs.push_back(num(coef));
      for (const E& f : keep) xs.push_back(f);
      for (const auto& pr : rads) xs.push_back(pow_e(num(pr.second), num(Rat(1, pr.first))));
      return mul_n(xs);
    }
  }
  if (out.empty()) return num(coef);
  if (!coef.is_one()) out.insert(out.begin(), num(coef));
  if (out.size() == 1) return out[0];
  std::sort(out.begin(), out.end(), [](const E& a, const E& b) { return cmp(a, b) < 0; });
  return raw(Kind::Mul, out);
}

inline E pow_e(const E& b_in, const E& e_in) {
  E b = simp(b_in), p = simp(e_in);
  if (is_num(p)) {
    if (p->num.is_zero()) return num(Rat(1));
    if (p->num.is_one()) return b;
    if (is_num(b) && p->num.is_int()) {
      const long long e = p->num.n;
      if (e > -32 && e < 32 && !(b->num.is_zero() && e < 0)) return num(rpow(b->num, e));
    }
    // **数の根は「有理化して、中身から完全冪を外に出す」形に正規化する**:
    //   sqrt(8) -> 2 sqrt(2)   sqrt(4) -> 2      sqrt(9/4) -> 3/2
    //   sqrt(1/2) -> sqrt(2)/2   2^(-1/2) -> sqrt(2)/2   sqrt(2/3) -> sqrt(6)/3
    // 教科書の答えは**分母に根号を残さない**ので、正規形の側で一度に片づける
    // （表示のときに直すやり方だと、同じ数が別の木のまま残って差が 0 にならない）。
    //
    // やり方: 指数 p/q を「整数部 k」と「0 < p'/q < 1」に分け、
    //   a^(p'/q) = (n/d)^(p'/q) = (n^p' * d^(q-p'))^(1/q) / d
    // と書き直す（分母が根の外に出る）。あとは中身から q 乗の因数を外に出すだけ。
    if (is_num(b) && !p->num.is_int() && !b->num.neg()) {
      if (b->num.is_zero()) { if (p->num.n > 0) return num(Rat(0)); }
      else {
        const long long q = p->num.d;
        if (q > 1 && q <= 8) {
          const long long pp = p->num.n;
          const long long k = pp >= 0 ? pp / q : -((-pp + q - 1) / q);   // 負のときは下へ切る
          const long long pf = pp - k * q;                               // 0 < pf < q
          long long M = 1;
          bool ok = true;
          for (long long i = 0; i < pf && ok; ++i) ok = mul_safe(M, b->num.n, M);
          for (long long i = 0; i < q - pf && ok; ++i) ok = mul_safe(M, b->num.d, M);
          if (ok) {
            long long s = 1, m = M;                        // M = s^q * m（m は q 乗の因数を持たない）
            for (long long f = 2; f * f <= m && f < 4096; ++f) {
              long long pw = 1;
              bool ovf = false;
              for (long long i = 0; i < q; ++i) { pw *= f; if (pw > m) { ovf = true; break; } }
              if (ovf) break;                              // これより大きい f では q 乗が中身を超える
              while (m % pw == 0) { m /= pw; s *= f; }
            }
            const Rat coef = rpow(b->num, k) * Rat(s, b->num.d);
            if (m == 1) return num(coef);
            const E rest = raw(Kind::Pow, {num(Rat(m)), num(Rat(1, q))});
            if (coef.is_one()) return rest;
            return mul_n({num(coef), rest});
          }
        }
      }
    }
  }
  // **a^(log_a M) = M**（指数と対数は逆の操作）。log(2, x) = log(2, 5) を解くと
  // x = 2^(log_2 5) が出るので、ここで畳まないと答えが 5 にならない。
  if (is_num(b) && b->num.n > 0) {
    Rat k(1);
    E core = p;
    if (p->k == Kind::Mul && p->kids.size() == 2 && is_num(p->kids[0])) {
      k = p->kids[0]->num;
      core = p->kids[1];
    }
    if (core->k == Kind::Fn && core->name == "log" && core->kids.size() == 2 &&
        is_num(core->kids[0]) && core->kids[0]->num == b->num)
      return pow_e(core->kids[1], num(k));
  }
  if (b->k == Kind::Pow) {                                      // (a^m)^n = a^(mn)
    const E& in = b->kids[1];
    // **内側が整数でないなら畳んでよい。** 内側が根（1/2 など）なら、底は実数の範囲で
    // 0 以上でなければ元の式が定義されないので、指数を掛けても意味が変わらない。
    // 逆に内側が整数で外側が分数のときは畳めない（sqrt(x^2) は |x| で、x ではない）。
    //
    // これが無いと `sqrt(y)/y` と `1/sqrt(y)` が**別の木のまま同じ印字**になり、
    // 差を取っても 0 にならない（selftest 2000 件で 2 件、この形で落ちた）。
    if (is_num(in) && is_num(p) && ((in->num.is_int() && p->num.is_int()) || !in->num.is_int()))
      return pow_e(b->kids[0], num(in->num * p->num));
  }
  if (b->k == Kind::Mul && is_num(p) && p->num.is_int()) {       // (ab)^n = a^n b^n
    std::vector<E> fs;
    for (const E& f : b->kids) fs.push_back(pow_e(f, p));
    return mul_n(fs);
  }
  return raw(Kind::Pow, {b, p});
}

// 最大公約数（負も受ける）
inline long long llgcd(long long a, long long b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b) { const long long t = a % b; a = b; b = t; }
  return a;
}

// 約数を**小さい順**に並べる（有理根定理で根を探す順を決めるのに使う）
inline void divisors(long long v, std::vector<long long>& out) {
  if (v < 0) v = -v;
  if (v == 0) { out.push_back(1); return; }
  for (long long d = 1; d * d <= v; ++d)
    if (v % d == 0) {
      out.push_back(d);
      if (d != v / d) out.push_back(v / d);
    }
  std::sort(out.begin(), out.end());
}

// v の k 乗根が整数なら true（対数の底を見つけるのに使う）
inline bool iroot(long long v, long long k, long long& out) {
  if (v < 0 || k <= 0) return false;
  if (v == 0) { out = 0; return true; }
  long long r = (long long)std::llround(std::pow((double)v, 1.0 / (double)k));
  for (long long c = r > 2 ? r - 2 : 0; c <= r + 2; ++c) {
    long long q = 1;
    bool ovf = false;
    for (long long i = 0; i < k; ++i) { q *= c; if (q > (1LL << 40)) { ovf = true; break; } }
    if (!ovf && q == v) { out = c; return true; }
  }
  return false;
}

// v = base^e と書けるうち **e が最大**のものを返す（4 は 2^2 であって 4^1 ではない、と見る）。
// これで log の底が一致するかを判定できる: log_4(8) は 4=2^2, 8=2^3 なので 3/2。
inline void prim_pow(const Rat& v, Rat& base, long long& e) {
  base = v;
  e = 1;
  if (v.n <= 0) return;
  for (long long k = 32; k >= 2; --k) {
    long long rn = 0, rd = 0;
    if (iroot(v.n, k, rn) && iroot(v.d, k, rd) && rn > 0) {
      base = Rat(rn, rd);
      e = k;
      return;
    }
  }
}

// log_a(x) が有理数になるか（log_2(8) = 3、log_4(8) = 3/2、log_2(1/8) = -3）
inline bool log_exact(const Rat& a, const Rat& x, Rat& out) {
  if (a.n <= 0 || x.n <= 0 || a.is_one()) return false;
  if (x.is_one()) { out = Rat(0); return true; }
  Rat ra, rx;
  long long ia = 0, ix = 0;
  prim_pow(a, ra, ia);
  prim_pow(x, rx, ix);
  if (ra == rx) { out = Rat(ix, ia); return true; }
  if (ra == Rat(1) / rx) { out = Rat(-ix, ia); return true; }
  return false;
}

inline bool is_pi(const E& e) {
  return e->k == Kind::Fn && e->name == "pi" && e->kids.empty();
}

// 角が c·π の形か（c は有理数）。0 も c = 0 として受ける
inline bool pi_coeff(const E& a, Rat& c) {
  if (is_num(a) && a->num.is_zero()) { c = Rat(0); return true; }
  if (is_pi(a)) { c = Rat(1); return true; }
  if (a->k == Kind::Mul && a->kids.size() == 2 && is_num(a->kids[0]) && is_pi(a->kids[1])) {
    c = a->kids[0]->num;
    return true;
  }
  return false;
}

// sin(t·π/12) の厳密値（t は 0..23）。15°(t が 12 と互いに素) は返さない
inline bool sin12(long long t, E& out) {
  const long long sign = t < 12 ? 1 : -1;
  long long u = t % 12;
  if (u > 6) u = 12 - u;                             // sin は 90° をはさんで対称
  E v;
  if (u == 0) v = num(Rat(0));
  else if (u == 2) v = num(Rat(1, 2));               // 30°
  else if (u == 3) v = mul_n({num(Rat(1, 2)), pow_e(num(Rat(2)), num(Rat(1, 2)))});   // 45°
  else if (u == 4) v = mul_n({num(Rat(1, 2)), pow_e(num(Rat(3)), num(Rat(1, 2)))});   // 60°
  else if (u == 6) v = num(Rat(1));                  // 90°
  else return false;                                 // 15° 刻みの半端な角は畳まない
  out = sign < 0 ? mul_n({num(Rat(-1)), v}) : v;
  return true;
}

// c·π の sin / cos / tan を厳密に返す（教科書の値の表）
inline bool trig_exact(const std::string& name, const Rat& c, E& out) {
  const long long d = c.d;
  long long nn = c.n % (2 * d);                      // c を [0, 2) に落とす
  if (nn < 0) nn += 2 * d;
  const Rat t12 = Rat(nn, d) * Rat(12);
  if (!t12.is_int()) return false;
  const long long t = t12.n;                         // 0..23（15° 単位）
  if (t % 12 == 1 || t % 12 == 5 || t % 12 == 7 || t % 12 == 11) return false;
  if (name == "sin") return sin12(t, out);
  if (name == "cos") return sin12((t + 6) % 24, out);
  if (name == "tan") {
    if (t == 6 || t == 18) return false;             // 90°・270° は定義されない
    E sn, cs;
    if (!sin12(t, sn) || !sin12((t + 6) % 24, cs)) return false;
    out = mul_n({sn, pow_e(cs, num(Rat(-1)))});
    return true;
  }
  return false;
}

// 関数。数値で閉じるものだけ畳む（sqrt(4)=2 など）。それ以外は Fn のまま。
inline E fn_e(const std::string& name, std::vector<E> args) {
  for (E& a : args) a = simp(a);
  if (name == "sqrt" && args.size() == 1) return pow_e(args[0], num(Rat(1, 2)));
  // 小学校の書き方を「1 つのテキスト」で表すための記法。値としては割り算と足し算に畳む
  // （絵のほうは ts::present_arith が書かれたとおりに描く）。
  //   frac(a,b)      縦の分数
  //   mixed(w,a,b)   帯分数（w + a/b）
  if (name == "frac" && args.size() == 2)
    return mul_n({args[0], pow_e(args[1], num(Rat(-1)))});
  if (name == "mixed" && args.size() == 3)
    return add_n({args[0], mul_n({args[1], pow_e(args[2], num(Rat(-1)))})});
  // **常用対数は底 10 を明示した形に直す**（log(x) と log(10, x) を別の木にしない）
  if (name == "log" && args.size() == 1) return fn_e("log", {num(Rat(10)), args[0]});
  if (name == "log" && args.size() == 2 && is_num(args[0]) && is_num(args[1])) {
    Rat v;
    if (log_exact(args[0]->num, args[1]->num, v)) return num(v);
  }
  // 特別角（30°・45°・60° とその仲間）の三角関数は厳密な値にする
  if (args.size() == 1 && (name == "sin" || name == "cos" || name == "tan")) {
    Rat c;
    E v;
    if (pi_coeff(args[0], c) && trig_exact(name, c, v)) return v;
  }
  // 階乗・順列・組合せ（数学 A）。**値が決まるものだけ畳む**（n! は n が 0..20 の整数のとき。
  // 21! は int64 に入らないので、そこは畳まずに残す）
  if (name == "fact" && args.size() == 1 && is_num(args[0]) && args[0]->num.is_int() &&
      args[0]->num.n >= 0 && args[0]->num.n <= 20) {
    Rat v(1);
    for (long long i = 2; i <= args[0]->num.n; ++i) v = v * Rat(i);
    return num(v);
  }
  if ((name == "P" || name == "C") && args.size() == 2 && is_num(args[0]) && is_num(args[1]) &&
      args[0]->num.is_int() && args[1]->num.is_int() && args[0]->num.n >= 0 &&
      args[1]->num.n >= 0 && args[1]->num.n <= args[0]->num.n && args[0]->num.n <= 62) {
    const long long n = args[0]->num.n, k = args[1]->num.n;
    Rat v(1);
    for (long long i = 0; i < k; ++i) v = v * Rat(n - i);       // nPr = n(n-1)...(n-k+1)
    if (name == "C")
      for (long long i = 2; i <= k; ++i) v = v / Rat(i);        // nCr = nPr / k!
    return num(v);
  }
  // exp と ln も逆の操作（exp(ln M) = M、ln(exp M) = M、log(a, a^M) = M）
  if (name == "exp" && args.size() == 1) {
    Rat k(1);
    E core = args[0];
    if (core->k == Kind::Mul && core->kids.size() == 2 && is_num(core->kids[0])) {
      k = core->kids[0]->num;
      core = core->kids[1];
    }
    if (core->k == Kind::Fn && core->name == "ln" && core->kids.size() == 1)
      return pow_e(core->kids[0], num(k));
  }
  if (name == "ln" && args.size() == 1 && args[0]->k == Kind::Fn && args[0]->name == "exp" &&
      args[0]->kids.size() == 1)
    return args[0]->kids[0];
  if (name == "log" && args.size() == 2 && is_num(args[0]) && args[1]->k == Kind::Pow &&
      is_num(args[1]->kids[0]) && args[1]->kids[0]->num == args[0]->num)
    return args[1]->kids[1];
  if (args.size() == 1 && is_num(args[0])) {
    const Rat& r = args[0]->num;
    if (name == "abs") return num(r.neg() ? -r : r);
    if (name == "ln" && r.is_one()) return num(Rat(0));
    if ((name == "sin" || name == "tan") && r.is_zero()) return num(Rat(0));
    if (name == "cos" && r.is_zero()) return num(Rat(1));
    if (name == "exp" && r.is_zero()) return num(Rat(1));
  }
  return raw(Kind::Fn, args, name);
}

inline E simp(const E& e) {
  switch (e->k) {
    case Kind::Num:
    case Kind::Sym:
      return e;
    case Kind::Add: return add_n(e->kids);
    case Kind::Mul: return mul_n(e->kids);
    case Kind::Pow: return pow_e(e->kids[0], e->kids[1]);
    case Kind::Fn: return fn_e(e->name, e->kids);
    case Kind::Rel: return raw(Kind::Rel, {simp(e->kids[0]), simp(e->kids[1])}, e->name);
    case Kind::Sys: {
      std::vector<E> ks;
      for (const E& c : e->kids) ks.push_back(simp(c));
      return raw(Kind::Sys, ks);
    }
  }
  return e;
}

// 使いやすい別名（呼ぶ側は simp 済みの木を受け取る）
inline E add(const E& a, const E& b) { return add_n({a, b}); }
inline E sub(const E& a, const E& b) { return add_n({a, mul_n({num(Rat(-1)), b})}); }
inline E mul(const E& a, const E& b) { return mul_n({a, b}); }
inline E div(const E& a, const E& b) { return mul_n({a, pow_e(b, num(Rat(-1)))}); }
inline E neg(const E& a) { return mul_n({num(Rat(-1)), a}); }

// 関係式。`op` は "=" "<" "<=" ">" ">=" のどれか。
inline E rel(const std::string& op, const E& a, const E& b) {
  return raw(Kind::Rel, {simp(a), simp(b)}, op);
}
inline E eq(const E& a, const E& b) { return rel("=", a, b); }
inline bool is_rel(const E& e) { return e->k == Kind::Rel; }
inline bool is_eq(const E& e) { return e->k == Kind::Rel && e->name == "="; }

// 不等号の向きを裏返す。**負の数を両辺にかける／割るときに必ず要る**（中学生が最も間違える所で、
// 手順表示でもここだけは理由を書く）。
inline std::string flip_op(const std::string& op) {
  if (op == "<") return ">";
  if (op == "<=") return ">=";
  if (op == ">") return "<";
  if (op == ">=") return "<=";
  return op;                                       // "=" は裏返しても "="
}

// 連立（関係式の列）。1 本だけならそのまま返す（Sys で包むと印字が変わってしまう）
inline E sys(const std::vector<E>& rels) {
  if (rels.size() == 1) return simp(rels[0]);
  std::vector<E> ks;
  for (const E& c : rels) ks.push_back(simp(c));
  return raw(Kind::Sys, ks);
}

// ---------------------------------------------------------------- 展開
//
// 分配法則を simp に入れない理由: (x+1)^100 のような式で爆発するし、「くくった形」を保ちたい
// 場面（因数分解の手順表示）で邪魔になる。展開が必要なのは
//   * 同類項をまとめて ax + b = 0 の形にするとき（solve）
//   * 人が期待する「答えの形」にするとき（eval）
// なので、そのときだけ明示的に呼ぶ。指数は整数かつ小さいものだけ展開する。
inline E expand(const E& e);

inline E expand_mul(std::vector<E> fs) {
  std::vector<E> sum{num(Rat(1))};                 // 部分積の集まり（=和の項）
  for (const E& f : fs) {
    std::vector<E> terms;
    if (f->k == Kind::Add) terms = f->kids;
    else terms.push_back(f);
    std::vector<E> next;
    next.reserve(sum.size() * terms.size());
    for (const E& a : sum)
      for (const E& b : terms) next.push_back(mul_n({a, b}));
    sum = std::move(next);
  }
  return add_n(sum);
}

inline E expand(const E& e) {
  switch (e->k) {
    case Kind::Num:
    case Kind::Sym:
      return e;
    case Kind::Add: {
      std::vector<E> xs;
      for (const E& c : e->kids) xs.push_back(expand(c));
      return add_n(xs);
    }
    case Kind::Mul: {
      std::vector<E> xs;
      for (const E& c : e->kids) xs.push_back(expand(c));
      return expand_mul(xs);
    }
    case Kind::Pow: {
      E b = expand(e->kids[0]);
      E p = expand(e->kids[1]);
      if (b->k == Kind::Add && is_num(p) && p->num.is_int() && p->num.n > 1 && p->num.n <= 8) {
        std::vector<E> fs((size_t)p->num.n, b);     // (a+b)^n を n 個の積に開く
        return expand_mul(fs);
      }
      return pow_e(b, p);
    }
    case Kind::Fn: {
      std::vector<E> xs;
      for (const E& c : e->kids) xs.push_back(expand(c));
      return fn_e(e->name, xs);
    }
    case Kind::Rel:
      return raw(Kind::Rel, {expand(e->kids[0]), expand(e->kids[1])}, e->name);
    case Kind::Sys: {
      std::vector<E> ks;
      for (const E& c : e->kids) ks.push_back(expand(c));
      return raw(Kind::Sys, ks);
    }
  }
  return e;
}

// ---------------------------------------------------------------- 印字

// 表示のための次数。内部の正規順序（cmp）は「同じ式か」を判定するためのもので、
// 読みやすさは別問題。人は x^2 - 5x + 6 の順で書くので、表示だけ次数の降順に並べ替える。
inline long long disp_degree(const E& e) {
  switch (e->k) {
    case Kind::Num: return 0;
    case Kind::Sym: return 1;
    case Kind::Pow: {
      const E& p = e->kids[1];
      if (is_num(p) && p->num.is_int()) return disp_degree(e->kids[0]) * p->num.n;
      return 1;
    }
    case Kind::Mul: {
      long long d = 0;
      for (const E& c : e->kids) d += disp_degree(c);
      return d;
    }
    case Kind::Add: {
      long long d = 0;
      for (const E& c : e->kids) d = std::max(d, disp_degree(c));
      return d;
    }
    default: return 1;
  }
}

// 項に出てくる変数のうち、辞書順で最初のもの（表示の並び替えに使う）
inline std::string disp_var(const E& e) {
  if (e->k == Kind::Sym) return e->name;
  std::string best;
  for (const E& c : e->kids) {
    const std::string v = disp_var(c);
    if (v.empty()) continue;
    if (best.empty() || v < best) best = v;
  }
  return best;
}

// 項の中の「変数 -> 次数」（表示の並び替え用。変数名の順に並べて返す）
inline void term_degs(const E& e, std::vector<std::pair<std::string, long long>>& out) {
  if (e->k == Kind::Sym) {
    for (std::pair<std::string, long long>& p : out)
      if (p.first == e->name) { p.second += 1; return; }
    out.push_back({e->name, 1});
    return;
  }
  if (e->k == Kind::Pow && is_sym(e->kids[0]) && is_num(e->kids[1]) &&
      e->kids[1]->num.is_int()) {
    const long long d = e->kids[1]->num.n;
    for (std::pair<std::string, long long>& p : out)
      if (p.first == e->kids[0]->name) { p.second += d; return; }
    out.push_back({e->kids[0]->name, d});
    return;
  }
  for (const E& c : e->kids) term_degs(c, out);
}

inline long long deg_of(const std::vector<std::pair<std::string, long long>>& v,
                       const std::string& name) {
  for (const std::pair<std::string, long long>& p : v)
    if (p.first == name) return p.second;
  return 0;
}

// 和の項を表示順に並べたもの。
//
//   1. 次数の降順（人は x^2 - 5x + 6 の順で書く）
//   2. 同じ次数なら、**変数ごとの次数を変数名の順に見て、次数の大きい方を先**にする
//      （教科書は x^2 + 12xy + 36y^2 の順。正規順序だけだと係数の大小で並んで
//        `2x - y` が `-y + 2*x` と出る。実写の `(2x - y)^2` を読んだときに出た）
//   3. それでも決まらなければ正規順序（決定的にするため）
inline std::vector<E> disp_terms(const E& e) {
  std::vector<E> ts = e->kids;
  std::stable_sort(ts.begin(), ts.end(), [](const E& a, const E& b) {
    const long long da = disp_degree(a), db = disp_degree(b);
    if (da != db) return da > db;
    std::vector<std::pair<std::string, long long>> va, vb;
    term_degs(a, va);
    term_degs(b, vb);
    std::vector<std::string> names;
    for (const std::pair<std::string, long long>& p : va) names.push_back(p.first);
    for (const std::pair<std::string, long long>& p : vb) names.push_back(p.first);
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    for (const std::string& n : names) {
      const long long ea = deg_of(va, n), eb = deg_of(vb, n);
      if (ea != eb) return ea > eb;
    }
    return cmp(a, b) < 0;
  });
  return ts;
}

// 演算子の優先順位。括弧を出しすぎず、足りなくもしないため
inline int prec(const E& e) {
  switch (e->k) {
    case Kind::Sys: return -1;
    case Kind::Rel: return 0;
    case Kind::Add: return 1;
    case Kind::Mul: return 2;
    case Kind::Pow: return 3;
    default: return 4;
  }
}

// 積を「分子の因子」と「分母の因子」に分ける。**印字のためだけ**の処理で、意味の木は変えない。
// 組版側（typeset.hpp の split_frac）と同じ考え方: 指数が負のものと、有理数の分母が分母に行く。
// これをやらないと 2/(3x) が "2/3*1/x" と出る（読めない）。
inline void split_num_den(const E& e, std::vector<E>& up, std::vector<E>& down) {
  std::vector<E> fs;
  if (e->k == Kind::Mul) fs = e->kids; else fs.push_back(e);
  for (const E& f : fs) {
    if (f->k == Kind::Pow && is_num(f->kids[1]) && f->kids[1]->num.neg()) {
      const Rat q = f->kids[1]->num;
      down.push_back(q.n == -1 && q.d == 1 ? f->kids[0] : pow_e(f->kids[0], num(-q)));
    } else if (is_num(f) && !f->num.is_int()) {
      if (f->num.n != 1) up.push_back(num(Rat(f->num.n)));
      down.push_back(num(Rat(f->num.d)));
    } else {
      up.push_back(f);
    }
  }
}

inline std::string to_infix(const E& e, int parent = 0);

inline std::string wrap(const E& e, int p) {
  const std::string s = to_infix(e, p);
  return prec(e) < p ? "(" + s + ")" : s;
}

inline std::string to_infix(const E& e, int parent) {
  (void)parent;
  switch (e->k) {
    case Kind::Num: return e->num.str();
    case Kind::Sym: return e->name;
    case Kind::Add: {
      std::string s;
      const std::vector<E> ts = disp_terms(e);
      for (size_t i = 0; i < ts.size(); ++i) {
        const E& t = ts[i];
        // 負の項は "+ -3" ではなく "- 3" と書く
        Rat c; E rest;
        split_coeff(t, c, rest);
        const bool minus = c.neg();
        if (i == 0) { if (minus) s += "-"; }
        else s += minus ? " - " : " + ";
        Rat ac = minus ? -c : c;
        E body = ac.is_one() && !is_num(rest) ? rest
                 : (is_num(rest) && rest->num.is_one() ? num(ac) : mul_n({num(ac), rest}));
        // **中身が和なら括弧が要る**。a - (2x + 1) を "a - 2*x + 1" と書くと符号が変わる
        // （展開すると和の項に割れるので、--no-expand の道でだけ出ていた）
        s += wrap(body, 2);
      }
      return s;
    }
    case Kind::Mul: {
      {
        std::vector<E> up, down;
        split_num_den(e, up, down);
        if (!down.empty()) {
          const E nu = up.empty() ? num(Rat(1)) : (up.size() == 1 ? up[0] : mul_n(up));
          const E de = down.size() == 1 ? down[0] : mul_n(down);
          return wrap(nu, 2) + "/" + wrap(de, 3);
        }
      }
      // 係数 -1 は "-1*x" ではなく "-x" と書く（人はそう書く）
      std::string s;
      size_t start = 0;
      if (is_num(e->kids[0]) && e->kids[0]->num.n == -1 && e->kids[0]->num.d == 1 &&
          e->kids.size() > 1) {
        s += "-";
        start = 1;
      }
      for (size_t i = start; i < e->kids.size(); ++i) {
        if (i > start) s += "*";
        s += wrap(e->kids[i], 2);
      }
      return s;
    }
    case Kind::Pow: {
      const E& b = e->kids[0];
      const E& p = e->kids[1];
      if (is_num(p) && p->num == Rat(1, 2)) return "sqrt(" + to_infix(b) + ")";
      // 負の指数は分数で書く（LaTeX 側と組版側は分数なのに、中置だけ x^(-1) と出ていた）
      if (is_num(p) && p->num.neg()) {
        const std::string den = p->num.n == -1 && p->num.d == 1
                                    ? wrap(b, 3)
                                    : wrap(pow_e(b, num(-p->num)), 3);
        return "1/" + den;
      }
      // 指数が整数でない数のときは括弧が必須。"2^1/2" は自分のパーサで (2^1)/2 に読めてしまう
      // ので、印字したものを読み直せなくなる（往復不変が壊れる）。
      const bool need = !is_num(p) || !p->num.is_int() || p->num.neg();
      const std::string ps = need ? "(" + to_infix(p) + ")" : to_infix(p);
      // **底が負の数や分数のときも括弧が要る**。(1/2)^n を "1/2^(n)" と書くと 1/(2^n) に、
      // (-2)^n を "-2^(n)" と書くと -(2^n) に読み戻ってしまう（等比数列の公比でよく出る）。
      const bool bneed = is_num(b) && (b->num.neg() || !b->num.is_int());
      return (bneed ? "(" + to_infix(b) + ")" : wrap(b, 4)) + "^" + ps;
    }
    case Kind::Fn: {
      if (e->kids.empty()) return e->name;            // 定数（pi）は括弧を付けない
      std::string s = e->name + "(";
      for (size_t i = 0; i < e->kids.size(); ++i) { if (i) s += ", "; s += to_infix(e->kids[i]); }
      return s + ")";
    }
    case Kind::Rel:
      return to_infix(e->kids[0]) + " " + e->name + " " + to_infix(e->kids[1]);
    case Kind::Sys: {
      // 連立は ", " で並べる。**この形をパーサが読み戻せる**（往復不変を Sys でも保つ）
      std::string s;
      for (size_t i = 0; i < e->kids.size(); ++i) {
        if (i) s += ", ";
        s += to_infix(e->kids[i]);
      }
      return s;
    }
  }
  return "?";
}

// LaTeX。デモの表示と、学習データを描くレンダラの入力に使う
inline std::string to_latex(const E& e, int parent = 0) {
  switch (e->k) {
    case Kind::Num: {
      if (e->num.is_int()) return e->num.str();
      // **負の分数はマイナスを前に出す**（-1/2 は \frac{-1}{2} ではなく -\frac{1}{2}）
      const bool ng = e->num.neg();
      const std::string f = "\\frac{" + std::to_string(ng ? -e->num.n : e->num.n) + "}{" +
                            std::to_string(e->num.d) + "}";
      return ng ? "-" + f : f;
    }
    case Kind::Sym: return e->name;
    case Kind::Add: {
      std::string s;
      const std::vector<E> ts = disp_terms(e);
      for (size_t i = 0; i < ts.size(); ++i) {
        Rat c; E rest;
        split_coeff(ts[i], c, rest);
        const bool minus = c.neg();
        if (i == 0) { if (minus) s += "-"; }
        else s += minus ? " - " : " + ";
        Rat ac = minus ? -c : c;
        E body = ac.is_one() && !is_num(rest) ? rest
                 : (is_num(rest) && rest->num.is_one() ? num(ac) : mul_n({num(ac), rest}));
        s += body->k == Kind::Add ? "(" + to_latex(body) + ")" : to_latex(body, 1);
      }
      return s;
    }
    case Kind::Mul: {
      // 有理数の係数は分数として前に出す（\frac{2}{3}x のように）
      {
        std::vector<E> up, down;
        split_num_den(e, up, down);
        if (!down.empty()) {
          const E nu = up.empty() ? num(Rat(1)) : (up.size() == 1 ? up[0] : mul_n(up));
          const E de = down.size() == 1 ? down[0] : mul_n(down);
          return "\\frac{" + to_latex(nu) + "}{" + to_latex(de) + "}";
        }
      }
      std::string s;
      size_t start = 0;
      if (is_num(e->kids[0]) && e->kids[0]->num.n == -1 && e->kids[0]->num.d == 1 &&
          e->kids.size() > 1) {
        s += "-";
        start = 1;
      }
      for (size_t i = start; i < e->kids.size(); ++i) {
        const E& f = e->kids[i];
        std::string t = to_latex(f, 2);
        if (f->k == Kind::Add) t = "(" + t + ")";
        s += (i > start ? " " : "") + t;
      }
      return s;
    }
    case Kind::Pow: {
      const E& b = e->kids[0];
      const E& p = e->kids[1];
      if (is_num(p) && p->num == Rat(1, 2)) return "\\sqrt{" + to_latex(b) + "}";
      // 負の指数は分数で書く（-1 だけでなく -1/2 なども。往復で表示が変わらないように）
      if (is_num(p) && p->num.neg())
        return "\\frac{1}{" +
               to_latex(p->num.n == -1 && p->num.d == 1 ? b : pow_e(b, num(-p->num))) + "}";
      std::string bs = to_latex(b, 4);
      if (b->k == Kind::Add || b->k == Kind::Mul || b->k == Kind::Pow ||
          (is_num(b) && (b->num.neg() || !b->num.is_int())))
        bs = "(" + bs + ")";                           // (-2)^n を -2^n と書かない
      return bs + "^{" + to_latex(p) + "}";
    }
    case Kind::Fn: {
      if (e->kids.empty()) return "\\" + e->name;    // \pi
      if (e->name == "fact" && e->kids.size() == 1) return to_latex(e->kids[0], 4) + "!";
      if ((e->name == "P" || e->name == "C") && e->kids.size() == 2)
        return "{}_{" + to_latex(e->kids[0]) + "}" + e->name + "_{" + to_latex(e->kids[1]) + "}";
      // log は底を下付きで書く（log(a, x) -> \log_{a} x）
      if (e->name == "log" && e->kids.size() == 2)
        return "\\log_{" + to_latex(e->kids[0]) + "} " + to_latex(e->kids[1], 4);
      // Σ は sum(k, 1, n, 中身) の 4 引数で持ち、印字だけ数学の形にする
      if (e->name == "sum" && e->kids.size() == 4)
        return "\\sum_{" + to_latex(e->kids[0]) + "=" + to_latex(e->kids[1]) + "}^{" +
               to_latex(e->kids[2]) + "} " + to_latex(e->kids[3], 4);
      std::string s = "\\" + e->name + "(";
      for (size_t i = 0; i < e->kids.size(); ++i) { if (i) s += ", "; s += to_latex(e->kids[i]); }
      return s + ")";
    }
    case Kind::Rel: {
      const std::string op = e->name == "<=" ? "\\le" : e->name == ">=" ? "\\ge" : e->name;
      return to_latex(e->kids[0]) + " " + op + " " + to_latex(e->kids[1]);
    }
    case Kind::Sys: {
      std::string s = "\\begin{cases}";
      for (size_t i = 0; i < e->kids.size(); ++i) {
        if (i) s += " \\\\ ";
        else s += " ";
        s += to_latex(e->kids[i]);
      }
      return s + " \\end{cases}";
    }
  }
  (void)parent;
  return "?";
}

// ---------------------------------------------------------------- 構文解析
//
// 認識器ができるまでの入口であり、テストの入口でもある。暗黙の掛け算（5x, 2(x+1)）を
// 受けるのは、写真から読んだ式がそう書かれているから。

// 関数として扱う名前（これ以外の名前 + 括弧は掛け算）
inline bool is_fn_name(const std::string& n) {
  return n == "sqrt" || n == "frac" || n == "mixed" || n == "sin" || n == "cos" || n == "tan" || n == "ln" || n == "exp" ||
         n == "abs" || n == "sum" || n == "log" || n == "fact" || n == "P" || n == "C";
}

struct Parser {
  std::string s;
  size_t i = 0;
  std::string err;
  // **畳まないで読む**（小学校の計算の手順を出すため。arith.hpp を参照）。
  // raw のときは + - × ÷ ^ を Fn の "op_*" として残すので、正規化に巻き込まれない。
  bool raw = false;

  explicit Parser(std::string src) : s(std::move(src)) {}

  // 演算 1 つを作る。**raw と通常で道を分けない**（分けると片方だけ直す事故が起きる）
  E bin(const char* op, const E& a, const E& b) const {
    if (raw) return ex::raw(Kind::Fn, {a, b}, op);
    const std::string o = op;
    if (o == "op_add") return add_n({a, b});
    if (o == "op_sub") return add_n({a, mul_n({num(Rat(-1)), b})});
    if (o == "op_mul") return mul_n({a, b});
    if (o == "op_div") return mul_n({a, pow_e(b, num(Rat(-1)))});
    if (o == "op_pow") return pow_e(a, b);
    return a;
  }
  E un(const char* op, const E& a) const {
    if (raw) return ex::raw(Kind::Fn, {a}, op);
    return mul_n({num(Rat(-1)), a});                 // op_neg
  }

  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i; }
  bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
  bool peek(char c) { ws(); return i < s.size() && s[i] == c; }

  E parse_all() {
    std::vector<E> rels{parse_rel()};
    for (;;) {
      ws();
      if (peek(',') || peek(';')) {                  // 連立は "," か ";" で区切る
        if (!eat(',')) eat(';');
        rels.push_back(parse_rel());
        continue;
      }
      break;
    }
    ws();
    if (i < s.size() && err.empty()) err = "余分な文字: " + s.substr(i);
    if (rels.size() == 1) return rels[0];
    return ex::raw(Kind::Sys, rels);
  }
  // 関係演算子は 1 段だけ（a < b < c のような連鎖は受けない。中学の書き方に無いので、
  // 受けると「読めているつもりで別の意味」になる）
  E parse_rel() {
    E l = parse_add();
    ws();
    if (i + 1 < s.size() && (s[i] == '<' || s[i] == '>') && s[i + 1] == '=') {
      const std::string op = std::string(1, s[i]) + "=";
      i += 2;
      return ex::raw(Kind::Rel, {l, parse_add()}, op);
    }
    if (eat('<')) return ex::raw(Kind::Rel, {l, parse_add()}, "<");
    if (eat('>')) return ex::raw(Kind::Rel, {l, parse_add()}, ">");
    if (eat('=')) return ex::raw(Kind::Rel, {l, parse_add()}, "=");
    return l;
  }
  E parse_add() {
    E l = parse_mul();
    for (;;) {
      ws();
      if (eat('+')) l = bin("op_add", l, parse_mul());
      else if (eat('-')) l = bin("op_sub", l, parse_mul());
      else return l;
    }
  }
  // 小学校の計算では × と ÷ が字として書かれる（UTF-8 で 2 バイト）。読めるようにしておく
  bool eat_utf8(const char* seq) {
    ws();
    const size_t n = strlen(seq);
    if (s.compare(i, n, seq) == 0) { i += n; return true; }
    return false;
  }

  E parse_mul() {
    E l = parse_unary();
    for (;;) {
      ws();
      if (eat('*') || eat_utf8("\xc3\x97")) l = bin("op_mul", l, parse_unary());     // * ×
      else if (eat('/') || eat_utf8("\xc3\xb7"))                                     // / ÷
        l = bin("op_div", l, parse_unary());
      else if (i < s.size() && (isalpha((unsigned char)s[i]) || s[i] == '(' || s[i] == '{' ||
                                isdigit((unsigned char)s[i]))) {
        // 暗黙の掛け算。ただし数のあとに数が来る形（"2 3"）は書き間違いとして扱う
        if (isdigit((unsigned char)s[i]) && is_num(l)) { err = "数が続いています"; return l; }
        l = bin("op_mul", l, parse_unary());
      } else return l;
    }
  }
  E parse_unary() {
    ws();
    if (eat('-')) return un("op_neg", parse_unary());
    if (eat('+')) return parse_unary();
    return parse_pow();
  }
  E parse_pow() {
    E b = parse_atom();
    ws();
    // **後置の階乗**（5! と書く）。^ より先に付く（3!^2 は (3!)^2）
    while (peek('!')) {
      eat('!');
      b = raw ? un("op_fact", b) : fn_e("fact", {b});
      ws();
    }
    if (eat('^')) return bin("op_pow", b, parse_unary());   // 右結合
    return b;
  }
  E parse_atom() {
    ws();
    if (i >= s.size()) { err = "式が途中で終わっています"; return num(Rat(0)); }
    if (eat('(')) {
      E e = parse_add();
      if (!eat(')')) err = "閉じ括弧がありません";
      return e;
    }
    if (eat('{')) {                                  // 小学校の計算は { } も使う
      E e = parse_add();
      if (!eat('}')) err = "閉じ中括弧がありません";
      return e;
    }
    if (isdigit((unsigned char)s[i])) {
      long long v = 0;
      while (i < s.size() && isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0');
      if (i < s.size() && s[i] == '.') {              // 小数は有理数に直す（0.25 -> 1/4）
        ++i;
        long long f = 0, den = 1;
        while (i < s.size() && isdigit((unsigned char)s[i])) { f = f * 10 + (s[i++] - '0'); den *= 10; }
        return num(Rat(v * den + f, den));
      }
      return num(Rat(v));
    }
    // **|x - 1| の書き方**（教科書はこう書く）。abs(...) と同じ木にする
    if (peek('|')) {
      eat('|');
      const E a = parse_add();
      if (!eat('|')) err = "絶対値の | が閉じていません";
      return raw ? un("op_abs", a) : fn_e("abs", {a});
    }
    if (isalpha((unsigned char)s[i])) {
      // 名前は**英字の連なりだけ**（数字は含めない。`x2` は x*2 と読む）
      std::string name;
      while (i < s.size() && isalpha((unsigned char)s[i])) name += s[i++];
      // **円周率は定数**（p*i と読ませない）。2pi や pi/6 は掛け算・割り算として続く
      if (name == "pi") return fn_e("pi", {});
      // **関数呼び出しは名前が関数のときだけ**。そうしないと `2x(x - 1)` の `x(...)` が
      // 「関数 x の呼び出し」になり、`2x(x-1) = 5` が展開できない式として残る
      // （実写の写真でこの形が出て、答えが出せなかった）。数式では変数のあとの括弧は掛け算。
      if (peek('(') && is_fn_name(name)) {
        eat('(');
        std::vector<E> args{parse_add()};
        while (eat(',')) args.push_back(parse_add());
        if (!eat(')')) err = "関数の閉じ括弧がありません";
        // raw のとき:
        //   frac(a,b) は**1 つの数**として畳む（縦の分数は「割り算をする所」ではない）
        //   mixed(w,a,b) は op_mixed のまま残す（「帯分数を仮分数に直す」を手順に出す）
        if (raw && name == "frac" && args.size() == 2) {
          if (is_num(args[0]) && is_num(args[1]) && !args[1]->num.is_zero())
            return num(args[0]->num / args[1]->num);
          return bin("op_div", args[0], args[1]);
        }
        if (raw && name == "mixed" && args.size() == 3)
          return ex::raw(Kind::Fn, {args[0], args[1], args[2]}, "op_mixed");
        return fn_e(name, args);
      }
      // **英字が続いたら 1 文字ずつの変数の積**（`12xy` は 12*x*y。教科書はそう書く）。
      // 1 つの変数 "xy" にしていたら、次数が 1 と数えられて表示順も展開も狂った。
      if (name.size() > 1) {
        std::vector<E> fs;
        for (size_t j = 0; j + 1 < name.size(); ++j) fs.push_back(sym(std::string(1, name[j])));
        // **指数は最後の 1 文字にだけ付く**。`xy^2` は x·y^2 であって (xy)^2 ではない
        // （教科書の書き方。ここを間違えると 9xy^2 が 9x^2y^2 になる）
        E last = sym(std::string(1, name.back()));
        ws();
        if (peek('^')) {
          eat('^');
          last = bin("op_pow", last, parse_unary());
        }
        fs.push_back(last);
        return mul_n(fs);
      }
      return sym(name);
    }
    err = std::string("読めない文字: ") + s[i];
    ++i;
    return num(Rat(0));
  }
};

// 失敗したら why に理由が入り、戻り値は 0 になる
inline E parse(const std::string& src, std::string* why = nullptr) {
  Parser p(src);
  E e = p.parse_all();
  if (why) *why = p.err;
  return simp(e);
}

// **畳まないで読む**（小学校の計算の手順を出すため。ar::eval_steps に渡す）。
// simp を通さないので、書かれた順・書かれた括弧がそのまま木に残る。
inline E parse_raw(const std::string& src, std::string* why = nullptr) {
  Parser p(src);
  p.raw = true;
  E e = p.parse_all();
  if (why) *why = p.err;
  return e;
}

// ---------------------------------------------------------------- 補助

// 出現する変数（並びは決定的）
inline void collect_syms(const E& e, std::vector<std::string>& out) {
  if (e->k == Kind::Sym) {
    if (std::find(out.begin(), out.end(), e->name) == out.end()) out.push_back(e->name);
  }
  for (const E& c : e->kids) collect_syms(c, out);
}

// 有理数を小数で書く（**割り切れるときだけ**）。割り切れないときは空文字を返す。
// 小学校の計算は答えを小数で書くので、`261/10` ではなく `26.1` も見せたい。
// 循環小数を勝手に丸めると嘘になるので、10 の冪で割り切れる場合に限る。
inline std::string to_decimal(const Rat& r) {
  long long d = r.d;
  int digits = 0;
  while (d % 2 == 0 && digits < 12) { d /= 2; ++digits; }
  while (d % 5 == 0 && digits < 12) { d /= 5; ++digits; }
  if (d != 1) return "";                             // 3 や 7 が残る = 割り切れない
  long long den = 1, k = 0;
  while (den % r.d != 0 && k < 12) { den *= 10; ++k; }
  if (den % r.d != 0) return "";
  const long long scaled = r.n * (den / r.d);        // 小数点を右に k 桁ずらした整数
  const bool neg = scaled < 0;
  std::string s = std::to_string(neg ? -scaled : scaled);
  if (k > 0) {
    while ((long long)s.size() <= k) s.insert(s.begin(), '0');
    s.insert(s.end() - k, '.');
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
  }
  return (neg ? "-" : "") + s;
}

// 変数を式で置き換える（代入法、連立の後半、将来の微分の合成に使う）。
// 置き換えた後は simp を通すので、結果は正規形。
inline E subst(const E& e, const std::string& var, const E& val) {
  if (e->k == Kind::Sym) return e->name == var ? val : e;
  if (e->kids.empty()) return e;
  std::vector<E> ks;
  for (const E& c : e->kids) ks.push_back(subst(c, var, val));
  return simp(raw(e->k, ks, e->name));
}

// 数値評価（表示の最後だけで使う。厳密に閉じない sqrt などのため）
inline double approx(const E& e) {
  switch (e->k) {
    case Kind::Num: return e->num.f();
    case Kind::Sym: return 0.0;                        // 未知数は 0 とみなす（呼ぶ側で弾く）
    case Kind::Add: { double s = 0; for (const E& c : e->kids) s += approx(c); return s; }
    case Kind::Mul: { double s = 1; for (const E& c : e->kids) s *= approx(c); return s; }
    case Kind::Pow: return std::pow(approx(e->kids[0]), approx(e->kids[1]));
    case Kind::Fn: {
      if (e->name == "pi") return 3.14159265358979323846;
      if (e->name == "log" && e->kids.size() == 2)
        return std::log(approx(e->kids[1])) / std::log(approx(e->kids[0]));
      const double a = e->kids.empty() ? 0.0 : approx(e->kids[0]);
      if (e->name == "sin") return std::sin(a);
      if (e->name == "cos") return std::cos(a);
      if (e->name == "tan") return std::tan(a);
      if (e->name == "ln") return std::log(a);
      if (e->name == "exp") return std::exp(a);
      if (e->name == "abs") return std::fabs(a);
      return a;
    }
    case Kind::Rel: return approx(e->kids[0]) - approx(e->kids[1]);
    case Kind::Sys: return 0.0;                        // 連立に数値はない（呼ぶ側で弾く）
  }
  return 0.0;
}

}  // namespace ex
