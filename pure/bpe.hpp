// pix2tex の tokenizer.json を読む（byte-level BPE）— tools/bpe.py の鏡。
//
// **byte-level とは**: 文字ではなくバイトを単位にして、0..255 を「見える文字」に写してから
// BPE をかける方式（GPT-2 と同じ）。空白は U+0120 'Ġ' になる。戻すときは逆写像で
// バイトに戻す。だから未知の文字が原理的に出ない代わりに、id と文字の対応は人には読めない。
//
// json は 1175 語 + 1073 行の merges しか要らないので、必要な形だけ読む小さな読み取り器を
// 置いてある（外の json 実装を持ち込まないため。この repo は依存を増やさない）。
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace bpe {

// ---------------------------------------------------------------- 小さな json 読み

namespace js {

struct Val {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double num = 0;
  std::string s;
  std::vector<Val> arr;
  std::vector<std::pair<std::string, Val>> obj;

  const Val* get(const std::string& k) const {
    for (const auto& kv : obj)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

inline void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

inline void put_utf8(std::string& out, unsigned cp) {
  if (cp < 0x80) out.push_back((char)cp);
  else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

inline std::string parse_str(const std::string& s, size_t& i) {
  std::string out;
  ++i;                                                   // 開きの "
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      const char c = s[++i];
      ++i;
      switch (c) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'u': {
          unsigned cp = 0;
          for (int k = 0; k < 4 && i < s.size(); ++k, ++i) {
            const char h = s[i];
            cp = cp * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
          }
          put_utf8(out, cp);
          break;
        }
        default: out.push_back(c); break;                // \" \\ \/ はその字
      }
      continue;
    }
    out.push_back(s[i++]);
  }
  ++i;                                                   // 閉じの "
  return out;
}

inline Val parse(const std::string& s, size_t& i) {
  skip_ws(s, i);
  Val v;
  if (i >= s.size()) return v;
  const char c = s[i];
  if (c == '"') { v.t = Val::STR; v.s = parse_str(s, i); return v; }
  if (c == '{') {
    v.t = Val::OBJ;
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') { ++i; return v; }
    while (i < s.size()) {
      skip_ws(s, i);
      const std::string k = parse_str(s, i);
      skip_ws(s, i);
      ++i;                                               // :
      v.obj.emplace_back(k, parse(s, i));
      skip_ws(s, i);
      if (i < s.size() && s[i] == ',') { ++i; continue; }
      if (i < s.size() && s[i] == '}') { ++i; break; }
      break;
    }
    return v;
  }
  if (c == '[') {
    v.t = Val::ARR;
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == ']') { ++i; return v; }
    while (i < s.size()) {
      v.arr.push_back(parse(s, i));
      skip_ws(s, i);
      if (i < s.size() && s[i] == ',') { ++i; continue; }
      if (i < s.size() && s[i] == ']') { ++i; break; }
      break;
    }
    return v;
  }
  if (c == 't' || c == 'f') {
    v.t = Val::BOOL;
    v.b = c == 't';
    i += (c == 't') ? 4 : 5;
    return v;
  }
  if (c == 'n') { i += 4; return v; }
  v.t = Val::NUM;
  size_t j = i;
  while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '-' || s[j] == '+' ||
                          s[j] == '.' || s[j] == 'e' || s[j] == 'E')) ++j;
  v.num = atof(s.substr(i, j - i).c_str());
  i = j;
  return v;
}

}  // namespace js

// ---------------------------------------------------------------- tokenizer

struct Tokenizer {
  std::map<std::string, int> vocab;
  std::vector<std::string> inv;
  std::map<std::pair<std::string, std::string>, int> ranks;
  std::map<unsigned, unsigned char> u2b;                 // 見える文字 -> バイト
  std::map<unsigned char, std::string> b2u;              // バイト -> 見える文字（UTF-8）
  std::vector<std::string> specials;
  int pad = 0, bos = 1, eos = 2;

  // GPT-2 の byte -> 見える文字。表示できないバイトを 256 以降にずらすだけ。
  void build_maps() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      bool found = false;
      for (int x : bs)
        if (x == b) { found = true; break; }
      if (!found) { bs.push_back(b); cs.push_back(256 + n++); }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
      std::string u;
      js::put_utf8(u, (unsigned)cs[i]);
      b2u[(unsigned char)bs[i]] = u;
      u2b[(unsigned)cs[i]] = (unsigned char)bs[i];
    }
  }

  bool load(const std::string& path, std::string* why = nullptr) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { if (why) *why = path + " が開けません"; return false; }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    size_t i = 0;
    const js::Val root = js::parse(s, i);
    const js::Val* model = root.get("model");
    if (!model) { if (why) *why = path + " に model がありません"; return false; }
    const js::Val* v = model->get("vocab");
    const js::Val* m = model->get("merges");
    if (!v || v->t != js::Val::OBJ) { if (why) *why = "vocab が読めません"; return false; }
    int maxid = 0;
    for (const auto& kv : v->obj) {
      const int id = (int)kv.second.num;
      vocab[kv.first] = id;
      if (id > maxid) maxid = id;
    }
    inv.assign((size_t)maxid + 1, std::string());
    for (const auto& kv : vocab) inv[(size_t)kv.second] = kv.first;
    if (m && m->t == js::Val::ARR) {
      for (size_t k = 0; k < m->arr.size(); ++k) {
        const std::string& line = m->arr[k].s;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        ranks[{line.substr(0, sp), line.substr(sp + 1)}] = (int)k;
      }
    }
    if (const js::Val* at = root.get("added_tokens")) {
      for (const js::Val& t : at->arr)
        if (const js::Val* c = t.get("content")) specials.push_back(c->s);
    }
    build_maps();
    if (vocab.count("[PAD]")) pad = vocab["[PAD]"];
    if (vocab.count("[BOS]")) bos = vocab["[BOS]"];
    if (vocab.count("[EOS]")) eos = vocab["[EOS]"];
    return true;
  }

  int size() const { return (int)vocab.size(); }

  // ---------------------------------------------------------- 戻す
  std::string decode(const std::vector<int>& ids) const {
    std::string joined;
    for (int id : ids)
      if (id >= 0 && id < (int)inv.size()) joined += inv[(size_t)id];
    for (const std::string& sp : specials) {             // [BOS] などは落とす
      size_t p;
      while ((p = joined.find(sp)) != std::string::npos) joined.erase(p, sp.size());
    }
    std::string out;
    for (size_t i = 0; i < joined.size();) {             // UTF-8 -> 符号位置 -> バイト
      unsigned cp = (unsigned char)joined[i];
      size_t len = 1;
      if (cp >= 0xF0) { cp &= 0x07; len = 4; }
      else if (cp >= 0xE0) { cp &= 0x0F; len = 3; }
      else if (cp >= 0xC0) { cp &= 0x1F; len = 2; }
      for (size_t k = 1; k < len && i + k < joined.size(); ++k)
        cp = (cp << 6) | ((unsigned char)joined[i + k] & 0x3F);
      i += len;
      auto it = u2b.find(cp);
      if (it != u2b.end()) out.push_back((char)it->second);
    }
    return out;
  }

  // ---------------------------------------------------------- 割る
  std::vector<std::string> bpe_of(const std::string& piece) const {
    std::vector<std::string> sym;
    for (size_t i = 0; i < piece.size();) {              // 1 符号位置ずつ
      size_t len = 1;
      const unsigned char c = (unsigned char)piece[i];
      if (c >= 0xF0) len = 4;
      else if (c >= 0xE0) len = 3;
      else if (c >= 0xC0) len = 2;
      sym.push_back(piece.substr(i, len));
      i += len;
    }
    while (sym.size() > 1) {
      int best = -1;
      size_t at = 0;
      for (size_t i = 0; i + 1 < sym.size(); ++i) {
        auto it = ranks.find({sym[i], sym[i + 1]});
        if (it != ranks.end() && (best < 0 || it->second < best)) { best = it->second; at = i; }
      }
      if (best < 0) break;
      sym[at] += sym[at + 1];
      sym.erase(sym.begin() + (long)at + 1);
    }
    return sym;
  }

  // LaTeX は ASCII なので「英字の並び / 数字の並び / それ以外 1 文字」＋先頭の空白で切る
  // （本家の ByteLevel 正規表現の、この用途で効く部分）。
  std::vector<int> encode(const std::string& text, bool add_bos = false,
                          bool add_eos = false) const {
    std::vector<int> out;
    if (add_bos) out.push_back(bos);
    const auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    const auto digit = [](char c) { return c >= '0' && c <= '9'; };
    size_t i = 0;
    while (i < text.size()) {
      std::string piece;
      if (text[i] == ' ') { piece = " "; ++i; if (i >= text.size()) { /* 末尾の空白 */ } }
      if (i < text.size()) {
        const size_t j = i;
        if (alpha(text[i])) { while (i < text.size() && alpha(text[i])) ++i; }
        else if (digit(text[i])) { while (i < text.size() && digit(text[i])) ++i; }
        else ++i;
        piece += text.substr(j, i - j);
      }
      std::string u;
      for (char ch : piece) {
        auto it = b2u.find((unsigned char)ch);
        if (it != b2u.end()) u += it->second;
      }
      for (const std::string& s : bpe_of(u)) {
        auto it = vocab.find(s);
        out.push_back(it == vocab.end() ? pad : it->second);
      }
    }
    if (add_eos) out.push_back(eos);
    return out;
  }
};

// 読んだ LaTeX の空白を詰める（本家 post_process と同じ働き）。
// **空白を残すのは英字と英字の間だけ**。`\ `（バックスラッシュ＋空白）は LaTeX の
// 空白命令なので触らない。
inline std::string post_process(const std::string& s) {
  const auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == ' ') {
      const char prev = out.empty() ? '\0' : out.back();
      const char next = i + 1 < s.size() ? s[i + 1] : '\0';
      if (prev == '\\') { out.push_back(' '); continue; }          // \  は残す
      if (alpha(prev) && alpha(next)) { out.push_back(' '); continue; }
      continue;                                                    // それ以外は詰める
    }
    out.push_back(s[i]);
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  size_t b = 0;
  while (b < out.size() && out[b] == ' ') ++b;
  return out.substr(b);
}

}  // namespace bpe
