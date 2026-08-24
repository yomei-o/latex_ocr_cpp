// LaTeX のトークナイザ — 文字列 <-> トークン列（id の並び）。
//
// このファイルが決めることが、後の全部の契約になる:
//
//   * モデルが出すのは**トークンの id の列**であって文字列ではない。語彙をここで固定する。
//   * `decode(encode(s)) == s` を保つ（往復不変）。ここが崩れると、正解率が
//     「モデルの誤り」なのか「文字列に戻すときの誤り」なのか分けられなくなる。
//   * 空白は**トークンにしない**。LaTeX の `x + 1` と `x+1` は同じ式で、絵も同じ。
//     入れると「見えない違い」で不正解が出る。戻すときに読みやすい位置に入れ直す。
//
// 語彙は「組版できるもの」に合わせてある（画像と LaTeX が食い違うと学習が壊れるため）。
// 実測: gen_expr の式 4000 本で 36 種、最長 21 トークン。余白を見て 64 まで許す。
#pragma once
#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace tok {

// 特別なトークン。**0 が padding** なのは、損失で無視する印を 0 にしておくと
// 実装を間違えにくいから（両言語とも同じ約束）。
enum : int { PAD = 0, BOS = 1, EOS = 2, UNK = 3, NSPECIAL = 4 };

// 語彙。**並びが id なので、足すときは必ず末尾**（既存の重みが読めなくなる）。
inline const std::vector<std::string>& vocab() {
  static const std::vector<std::string> v = {
      "<pad>", "<s>", "</s>", "<unk>",
      // 数字
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
      // 演算子と関係
      "+", "-", "=", "<", ">", "\\le", "\\ge", "\\times", "\\div", "\\pm",
      // 構造
      "{", "}", "(", ")", "[", "]", "^", "_", "\\frac", "\\sqrt", "\\left", "\\right",
      // 関数と定数
      "\\sin", "\\cos", "\\tan", "\\ln", "\\log", "\\exp", "\\pi", "\\infty",
      // 文字（gen_expr が使うもの + よく出るもの）
      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
      "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
      // ギリシャ文字（よく出るもの）
      "\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\mu", "\\sigma",
      // 区切り
      ",", ".", "!", "|", "/",
  };
  return v;
}

inline const std::map<std::string, int>& index() {
  static const std::map<std::string, int> m = [] {
    std::map<std::string, int> t;
    const std::vector<std::string>& v = vocab();
    for (size_t i = 0; i < v.size(); ++i) t[v[i]] = (int)i;
    return t;
  }();
  return m;
}

inline int size() { return (int)vocab().size(); }

inline int id_of(const std::string& s) {
  const std::map<std::string, int>& m = index();
  const std::map<std::string, int>::const_iterator it = m.find(s);
  return it == m.end() ? (int)UNK : it->second;
}

inline const std::string& text_of(int id) {
  static const std::string unk = "<unk>";
  const std::vector<std::string>& v = vocab();
  return (id >= 0 && id < (int)v.size()) ? v[(size_t)id] : unk;
}

// LaTeX の文字列を**トークンの文字列**に割る（id にはまだしない）。
// 規則は 3 つだけ:
//   1. `\` で始まったら、続く英字をまとめて 1 つ（`\frac`）。英字が無ければ次の 1 文字（`\{`）
//   2. 空白は捨てる
//   3. それ以外は 1 文字で 1 つ
inline std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '\\') {
      size_t j = i + 1;
      while (j < s.size() && isalpha((unsigned char)s[j])) ++j;
      if (j == i + 1) ++j;                              // `\{` のような 1 文字のもの
      out.push_back(s.substr(i, j - i));
      i = j;
      continue;
    }
    if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') { ++i; continue; }
    out.push_back(std::string(1, s[i]));
    ++i;
  }
  return out;
}

// 文字列 -> id の列。bos/eos を付けるかは呼ぶ側が決める
inline std::vector<int> encode(const std::string& s, bool bos = false, bool eos = false) {
  std::vector<int> out;
  if (bos) out.push_back(BOS);
  for (const std::string& t : split(s)) out.push_back(id_of(t));
  if (eos) out.push_back(EOS);
  return out;
}

// id の列 -> 文字列。**読みやすい位置に空白を入れ直す**。
// `\frac` のような命令のあとに英字が続くと `\fracx` になって別の命令に化けるので、
// そこだけは必ず 1 つ空ける（往復不変のために要る規則であって、見た目の好みではない）。
inline std::string decode(const std::vector<int>& ids, bool skip_special = true) {
  std::string out;
  std::string prev;
  for (int id : ids) {
    if (skip_special && (id == PAD || id == BOS || id == EOS)) continue;
    const std::string& t = text_of(id);
    if (!out.empty()) {
      const bool prev_cmd = prev.size() > 1 && prev[0] == '\\';
      const bool t_alpha = !t.empty() && (isalpha((unsigned char)t[0]) || t[0] == '\\');
      if (prev_cmd && t_alpha) out += ' ';             // `\frac x` が `\fracx` にならないように
    }
    out += t;
    prev = t;
  }
  return out;
}

// 往復で戻るか（テストと、学習データを作るときの検査に使う）
inline bool round_trips(const std::string& s) {
  const std::vector<int> ids = encode(s);
  for (int id : ids)
    if (id == UNK) return false;
  return decode(ids) == decode(encode(decode(ids)));
}

}  // namespace tok
