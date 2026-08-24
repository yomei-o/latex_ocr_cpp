"""pix2tex の tokenizer.json を読む（byte-level BPE）— pure/bpe.hpp の鏡。

本家は huggingface の `tokenizers` を使う。ここでは**同じ json を自前で読む**。理由は
pure/bpe.hpp と同じものを両言語で持つためで、外の実装に合わせるのではなく、
json（vocab 1175 語 + merges 1073 行）という 1 つの出どころに両方を合わせる。

byte-level とは: 文字ではなく**バイト**を単位にして、0..255 を「見える文字」に写してから
BPE をかける方式（GPT-2 と同じ）。空白は U+0120 'Ġ' になる。戻すときは逆写像でバイトに
戻して UTF-8 として読む。

  json の decoder が null なので、本家は decode の後に「空白を全部消して Ġ を空白にする」
  という後始末をしている。逆写像で戻せば同じ結果になる（そちらが本来の形）。
"""
import json


def bytes_to_unicode():
    """GPT-2 の byte -> 見える文字の対応表。**表示できないバイトを 256 以降にずらす**だけ。"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.vocab = d["model"]["vocab"]                       # token 文字列 -> id
        self.inv = {v: k for k, v in self.vocab.items()}
        merges = d["model"].get("merges", [])
        self.ranks = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ") if isinstance(m, str) else m
            self.ranks[(a, b)] = i
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.specials = {t["content"] for t in d.get("added_tokens", []) if t.get("special")}
        self.pad = self.vocab.get("[PAD]", 0)
        self.bos = self.vocab.get("[BOS]", 1)
        self.eos = self.vocab.get("[EOS]", 2)

    def size(self):
        return len(self.vocab)

    # ------------------------------------------------------------ 戻す
    def decode(self, ids):
        s = "".join(self.inv.get(int(i), "") for i in ids)
        for t in self.specials:
            s = s.replace(t, "")
        return bytes(self.u2b.get(ch, 0) for ch in s).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ 割る
    def _bpe(self, piece):
        """1 かたまりに merges を順位の低い順（= 早く覚えた順）に当てる。"""
        sym = list(piece)
        while len(sym) > 1:
            best, at = None, -1
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at < 0:
                break
            sym[at:at + 2] = [sym[at] + sym[at + 1]]
        return sym

    def encode(self, text, bos=False, eos=False):
        """LaTeX 文字列 -> id 列。

        本家の pre_tokenizer は ByteLevel（GPT-2 の正規表現で切る）。LaTeX は ASCII なので、
        ここでは「英字の並び / 数字の並び / それ以外 1 文字」＋先頭の空白、で切る。
        `\\frac` のような並びはこの切り方で本家と同じ塊になる。
        """
        out = [self.bos] if bos else []
        i, n = 0, len(text)
        pieces = []
        while i < n:
            j = i
            sp = ""
            if text[i] == " ":
                sp = " "
                i += 1
                if i >= n:
                    pieces.append(sp)
                    break
                j = i
            if text[i].isalpha():
                while i < n and text[i].isalpha():
                    i += 1
            elif text[i].isdigit():
                while i < n and text[i].isdigit():
                    i += 1
            else:
                i += 1
            pieces.append(sp + text[j:i])
        for p in pieces:
            u = "".join(self.b2u[b] for b in p.encode("utf-8"))
            for s in self._bpe(u):
                out.append(self.vocab.get(s, self.vocab.get("[PAD]", 0)))
        if eos:
            out.append(self.eos)
        return out


def post_process(s):
    r"""読んだ LaTeX の空白を詰める（本家 post_process と同じ働き）— pure/bpe.hpp の鏡。

    **空白を残すのは英字と英字の間だけ**。`\ `（バックスラッシュ＋空白）は LaTeX の
    空白命令なので触らない。本家は正規表現を収束するまで回すが、規則を書き下すと
    「隣り合う 2 文字を見るだけ」で足りる。
    """
    out = []
    for i, ch in enumerate(s):
        if ch == " ":
            prev = out[-1] if out else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if prev == "\\" or (prev.isalpha() and nxt.isalpha()):
                out.append(" ")
            continue
        out.append(ch)
    return "".join(out).strip()
