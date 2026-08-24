"""LaTeX のトークナイザ（Python 側）— pure/tok.hpp の鏡。

語彙の**並びが id** なので、片方だけ足すと学習した重みが読めなくなる。
両方に足して tools/parity/tok.py で突き合わせること。

  python tools/tok.py --latex "\\frac{1}{2} + x^{2}"
"""
import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PAD, BOS, EOS, UNK, NSPECIAL = 0, 1, 2, 3, 4

# **並びが id なので、足すときは必ず末尾**（pure/tok.hpp と 1 行ずつ同じ）
VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>",
    # 数字
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # 演算子と関係
    "+", "-", "=", "<", ">", "\\le", "\\ge", "\\times", "\\div", "\\pm",
    # 構造
    "{", "}", "(", ")", "[", "]", "^", "_", "\\frac", "\\sqrt", "\\left", "\\right",
    # 関数と定数
    "\\sin", "\\cos", "\\tan", "\\ln", "\\log", "\\exp", "\\pi", "\\infty",
    # 文字
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # ギリシャ文字
    "\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\mu", "\\sigma",
    # 区切り
    ",", ".", "!", "|", "/",
]
INDEX = {t: i for i, t in enumerate(VOCAB)}


def size():
    return len(VOCAB)


def id_of(s):
    return INDEX.get(s, UNK)


def text_of(i):
    return VOCAB[i] if 0 <= i < len(VOCAB) else "<unk>"


def split(s):
    """LaTeX の文字列を**トークンの文字列**に割る（規則は 3 つだけ。C++ の tok::split と同じ）。"""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            if j == i + 1:
                j += 1                                   # `\{` のような 1 文字のもの
            out.append(s[i:j])
            i = j
            continue
        if s[i] in " \t\n":
            i += 1
            continue
        out.append(s[i])
        i += 1
    return out


def encode(s, bos=False, eos=False):
    out = [BOS] if bos else []
    out += [id_of(t) for t in split(s)]
    if eos:
        out.append(EOS)
    return out


def decode(ids, skip_special=True):
    """id の列 -> 文字列。**読みやすい位置に空白を入れ直す**（C++ の tok::decode と同じ規則）。"""
    out, prev = "", ""
    for i in ids:
        if skip_special and i in (PAD, BOS, EOS):
            continue
        t = text_of(i)
        if out:
            prev_cmd = len(prev) > 1 and prev[0] == "\\"
            t_alpha = bool(t) and (t[0].isalpha() or t[0] == "\\")
            if prev_cmd and t_alpha:
                out += " "                               # `\frac x` が `\fracx` にならないように
        out += t
        prev = t
    return out


def round_trips(s):
    ids = encode(s)
    if UNK in ids:
        return False
    return decode(ids) == decode(encode(decode(ids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", default="")
    ap.add_argument("--vocab", action="store_true")
    a = ap.parse_args()
    if a.vocab or not a.latex:
        print("%d 種類" % size())
        print(" ".join("%d:%s" % (i, t) for i, t in enumerate(VOCAB)))
        return 0
    ids = encode(a.latex, bos=True, eos=True)
    print("tokens: %s" % " ".join(text_of(i) for i in ids))
    print("ids   : %s" % " ".join(str(i) for i in ids))
    print("decode: %s" % decode(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
