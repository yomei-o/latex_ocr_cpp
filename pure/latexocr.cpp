// latexocr — 数式の画像から LaTeX を読む（C++ 側の入口）。
//
//   latexocr vocab                                語彙を出す
//   latexocr tok --latex "\frac{1}{2}"            トークンに割って戻す
//   latexocr gen --out data/train --n 2000        学習データを作る（画像 + LaTeX）
//   latexocr init --out models/init.pt            重みを初期化して書き出す
//   latexocr train --data data/train --steps 200  学習する
//   latexocr infer --img x.png --model m.pt       1 枚読む
//   latexocr pix2tex --img x.png                  本家の学習済みモデルで読む
//   latexocr selftest                             トークナイザの往復と勾配の検算
#define STB_IMAGE_IMPLEMENTATION
#define STB_TRUETYPE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#include "third_party/stb_image.h"
#include "typeset_impl.hpp"   // ts::render の実体（stb の実装が要るので .cpp からだけ）
#include "data.hpp"
#include "model.hpp"
#include "optim.hpp"
#include "tok.hpp"
#include "bpe.hpp"
#include "pix2tex.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static std::string arg_of(int argc, char** argv, const char* key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}
// 親から順に作る（`data/train/images` は data も train も無いと作れない）
static void mkdir_p(const std::string& d) {
  std::string cur;
  for (size_t i = 0; i <= d.size(); ++i) {
    if (i == d.size() || d[i] == '/' || d[i] == '\\') {
      if (!cur.empty() && cur != "." && cur != "..") {
#ifdef _WIN32
        _mkdir(cur.c_str());
#else
        mkdir(cur.c_str(), 0755);
#endif
      }
      if (i < d.size()) cur += d[i];
      continue;
    }
    cur += d[i];
  }
}

// 共通の設定（コマンド行から）
static mdl::Cfg cfg_of(int argc, char** argv) {
  mdl::Cfg c;
  c.H = std::atoi(arg_of(argc, argv, "--h", "64").c_str());
  c.W = std::atoi(arg_of(argc, argv, "--w", "256").c_str());
  c.d = std::atoi(arg_of(argc, argv, "--dim", "128").c_str());
  c.heads = std::atoi(arg_of(argc, argv, "--heads", "4").c_str());
  c.layers = std::atoi(arg_of(argc, argv, "--layers", "2").c_str());
  c.ff = std::atoi(arg_of(argc, argv, "--ff", "256").c_str());
  c.max_len = std::atoi(arg_of(argc, argv, "--max-len", "48").c_str());
  return c;
}

static int cmd_vocab() {
  printf("%d 種類\n", tok::size());
  for (int i = 0; i < tok::size(); ++i) printf("%d:%s ", i, tok::text_of(i).c_str());
  printf("\n");
  return 0;
}

static int cmd_tok(int argc, char** argv) {
  const std::string s = arg_of(argc, argv, "--latex", "");
  if (s.empty()) {
    printf("usage: latexocr tok --latex \"\\frac{1}{2} + x^{2}\"\n");
    return 1;
  }
  const std::vector<int> ids = tok::encode(s, true, true);
  printf("tokens: ");
  for (int id : ids) printf("%s ", tok::text_of(id).c_str());
  printf("\nids   : ");
  for (int id : ids) printf("%d ", id);
  printf("\ndecode: %s\n", tok::decode(ids).c_str());
  return 0;
}

// labels.txt を読む
struct Item { std::string img; std::string latex; };
static std::vector<Item> read_labels(const std::string& dir) {
  std::vector<Item> out;
  FILE* f = fopen((dir + "/labels.txt").c_str(), "rb");
  if (!f) return out;
  char line[4096];
  while (fgets(line, sizeof line, f)) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    const size_t tab = s.find('\t');
    if (tab == std::string::npos) continue;
    Item it;
    it.img = dir + "/images/" + s.substr(0, tab);
    it.latex = s.substr(tab + 1);
    out.push_back(it);
  }
  fclose(f);
  return out;
}

// png を読んで [H*W] の 0..1（1 = 黒）にする
static bool load_gray(const std::string& path, int H, int W, std::vector<float>& out) {
  int w = 0, h = 0, ch = 0;
  unsigned char* im = stbi_load(path.c_str(), &w, &h, &ch, 1);
  if (!im) return false;
  out.assign((size_t)H * W, 0.f);
  for (int y = 0; y < H && y < h; ++y)
    for (int x = 0; x < W && x < w; ++x)
      out[(size_t)y * W + x] = (255.f - im[(size_t)y * w + x]) / 255.f;
  stbi_image_free(im);
  return true;
}

// 学習データを作る。images/%06d.png と labels.txt（1 行 = ファイル名 <TAB> LaTeX）
// 画像を 8bit グレイで読む。**透過つきの png は alpha のほうに字が入っている**ので、
// alpha が一定でないならそちらを使う（本家 utils.pad の convert('LA') と同じ判断）。
static bool load_img(const std::string& path, ip::Img& out) {
  int w = 0, h = 0, ch = 0;
  unsigned char* d = stbi_load(path.c_str(), &w, &h, &ch, 2);
  if (!d) return false;
  out.w = w;
  out.h = h;
  out.p.assign((size_t)w * h, 255);
  bool alpha_varies = false;
  const unsigned char a0 = d[1];
  for (size_t i = 0; i < (size_t)w * h; ++i)
    if (d[2 * i + 1] != a0) { alpha_varies = true; break; }
  for (size_t i = 0; i < (size_t)w * h; ++i)
    out.p[i] = alpha_varies ? (unsigned char)(255 - d[2 * i + 1]) : d[2 * i];
  stbi_image_free(d);
  return true;
}

static int cmd_gen(int argc, char** argv) {
  const std::string out = arg_of(argc, argv, "--out", "");
  const int n = std::atoi(arg_of(argc, argv, "--n", "1000").c_str());
  const uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1").c_str(), nullptr, 10);
  const std::string fontp = arg_of(argc, argv, "--font", "");
  const bool no_images = has_flag(argc, argv, "--no-images");
  const std::string excl = arg_of(argc, argv, "--exclude", "");
  const bool allow_dup = has_flag(argc, argv, "--allow-dup");
  if (out.empty()) {
    printf("usage: latexocr gen --out data/train --n 2000 [--seed 1] [--font path.ttf]\n");
    printf("       [--exclude data/train] [--allow-dup]\n");
    return 1;
  }
  mdl::Cfg mc = cfg_of(argc, argv);
  dat::Cfg dc;
  dc.W = mc.W;
  dc.H = mc.H;
  dc.max_len = mc.max_len;
  std::string why;
  ts::Font font;
  if (!ts::load_font(font, fontp, &why)) { printf("%s\n", why.c_str()); return 1; }
  ts::Style st;
  st.italic_vars = false;
  mkdir_p(out);
  mkdir_p(out + "/images");
  FILE* lab = fopen((out + "/labels.txt").c_str(), "wb");
  if (!lab) { printf("%s/labels.txt が書けません\n", out.c_str()); return 1; }
  // **同じ式を二度出さない**。式の作り方は木を振るだけなので、放っておくと短い式が
  // 何度も出る（20000 件で重複を許すと中身は 14449 種類しかなかった）。学習の中身が
  // 薄くなるうえ、val に train と同じ式が 33% 混ざって精度が水増しされる。
  // --exclude で train の labels.txt を渡せば、val は train と 1 つも被らない。
  std::set<std::string> seen;
  int n_excl = 0;
  if (!excl.empty()) {
    for (const Item& it : read_labels(excl)) {
      seen.insert(it.latex);
      ++n_excl;
    }
    if (!n_excl) { printf("%s が読めません\n", excl.c_str()); return 1; }
  }
  Rng rng(seed);
  int made = 0, tries = 0, longest = 0, dropped = 0;
  std::vector<unsigned char> png((size_t)dc.H * dc.W);
  while (made < n && tries < n * 40) {
    ++tries;
    dat::Sample s;
    if (!dat::make_one(font, nullptr, st, rng, dc, s)) continue;
    if (!allow_dup && !seen.insert(s.latex).second) { ++dropped; continue; }
    if (s.len > longest) longest = s.len;
    char name[64];
    snprintf(name, sizeof name, "%06d.png", made);
    if (!no_images) {
      for (size_t i = 0; i < png.size(); ++i)
        png[i] = (unsigned char)(255.0f - 255.0f * s.img[i]);       // 白地に黒字で保存
      stbi_write_png((out + "/images/" + name).c_str(), dc.W, dc.H, 1, png.data(), dc.W);
    }
    fprintf(lab, "%s\t%s\n", name, s.latex.c_str());
    ++made;
  }
  fclose(lab);
  printf("%s に %d 件（試した回数 %d、重複で捨てた %d、除外リスト %d 件、最長 %d トークン、%dx%d）\n",
         out.c_str(), made, tries, dropped, n_excl, longest, dc.W, dc.H);
  return 0;
}

static int cmd_init(int argc, char** argv) {
  const std::string out = arg_of(argc, argv, "--out", "");
  const uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  if (out.empty()) { printf("usage: latexocr init --out models/init.pt [--seed 1234]\n"); return 1; }
  const mdl::Cfg c = cfg_of(argc, argv);
  mdl::Params p;
  mdl::build(p, c);
  Rng rng(seed);
  mdl::init_params(p, rng);
  mkdir_p("models");
  mdl::save(p, out);
  printf("wrote %s: %zu tensor、%zu パラメータ（d %d, heads %d, layers %d, V %d）\n", out.c_str(),
         p.order.size(), p.count(), c.d, c.heads, c.layers, c.V());
  return 0;
}

// warmup してから cosine で落とす。transformer は**最初の数百 step が一番壊れやすい**
// （注意の softmax が飽和したまま大きい step を踏むと戻ってこない）。頭を小さく入って、
// 後半で細かく詰める。tools/train.py の lr_at と同じ式。
static float lr_at(int step, int steps, float lr, float lr_min, int warmup) {
  if (step < warmup) return lr * (float)(step + 1) / (float)(warmup > 0 ? warmup : 1);
  const int span = steps - warmup > 0 ? steps - warmup : 1;
  const double t = (double)(step - warmup) / (double)span;
  return lr_min + 0.5f * (lr - lr_min) * (float)(1.0 + cos(3.14159265358979323846 * (t < 1.0 ? t : 1.0)));
}

static int cmd_train(int argc, char** argv) {
  const std::string data = arg_of(argc, argv, "--data", "");
  const std::string init = arg_of(argc, argv, "--init", "");
  const std::string out = arg_of(argc, argv, "--export", "");
  const int steps = std::atoi(arg_of(argc, argv, "--steps", "100").c_str());
  const int batch = std::atoi(arg_of(argc, argv, "--batch", "4").c_str());
  const int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  const float lr = (float)atof(arg_of(argc, argv, "--lr", "3e-4").c_str());
  const float lr_min = (float)atof(arg_of(argc, argv, "--lr-min", "1e-5").c_str());
  const int warmup = std::atoi(arg_of(argc, argv, "--warmup", "200").c_str());
  const float clip = (float)atof(arg_of(argc, argv, "--clip", "1.0").c_str());
  const uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  const int log_every = std::atoi(arg_of(argc, argv, "--log-every", "10").c_str());
  if (data.empty()) {
    printf("usage: latexocr train --data data/train [--init models/init.pt] [--export m.pt]\n"
           "                     [--steps 100] [--batch 4] [--lr 3e-4]\n");
    return 1;
  }
  const mdl::Cfg c = cfg_of(argc, argv);
  std::vector<Item> items = read_labels(data);
  if (limit > 0 && (int)items.size() > limit) items.resize((size_t)limit);
  if (items.empty()) { printf("%s/labels.txt が読めません\n", data.c_str()); return 1; }
  mdl::Params p;
  mdl::build(p, c);
  Rng rng(seed);
  if (init.empty()) mdl::init_params(p, rng);
  else {
    std::string why;
    if (!mdl::load(p, init, &why)) { printf("%s: %s\n", init.c_str(), why.c_str()); return 1; }
  }
  printf("data %zu 件、パラメータ %zu、d %d heads %d layers %d、batch %d、%d step、lr %g\n",
         items.size(), p.count(), c.d, c.heads, c.layers, batch, steps, lr);
  std::vector<Tensor> ps = p.all();
  Adam opt(ps, lr, 0.9f, 0.999f, 1e-8f, 0.f, true);
  double first = 0, last = 0;
  for (int step = 0; step < steps; ++step) {
    opt.lr = lr_at(step, steps, lr, lr_min, warmup);
    opt.zero_grad();
    double loss_sum = 0;
    int used = 0;
    for (int b = 0; b < batch; ++b) {
      const Item& it = items[(size_t)rng.below((uint64_t)items.size())];
      std::vector<float> img;
      if (!load_gray(it.img, c.H, c.W, img)) continue;
      const std::vector<int> ids = tok::encode(it.latex, true, true);
      if ((int)ids.size() < 2 || (int)ids.size() > c.max_len) continue;
      // teacher forcing: 入力は BOS..(末尾の 1 つ前)、正解は 1 つずらした列
      const std::vector<int> in(ids.begin(), ids.end() - 1);
      const std::vector<int> tgt(ids.begin() + 1, ids.end());
      const Tensor enc = mdl::encode(mdl::img_tensor(img, c), p, c);
      const Tensor logits = mdl::decode(enc, in, p, c);
      const std::vector<float> mask((size_t)tgt.size(), 1.f);
      const Tensor loss = sq::ce_loss(logits, tgt, mask);
      backward(loss);
      loss_sum += loss->data[0];
      ++used;
      free_graph(loss);
    }
    if (!used) { printf("step %d: 使える件が無い\n", step); break; }
    for (Tensor& t : ps)
      for (float& g : t->grad) g /= (float)used;      // batch で平均
    const float gn = clip_grad_norm(ps, clip);
    opt.step();
    const double l = loss_sum / used;
    if (step == 0) first = l;
    last = l;
    if (log_every > 0 && (step % log_every == 0 || step == steps - 1))
      printf("step %d: loss %.4f |g| %.2f lr %.2e\n", step, l, gn, opt.lr);
    fflush(stdout);
  }
  printf("loss %.4f -> %.4f（%d step）\n", first, last, steps);
  if (!out.empty()) {
    mkdir_p("models");
    mdl::save(p, out);
    printf("wrote %s\n", out.c_str());
  }
  return 0;
}

static int cmd_infer(int argc, char** argv) {
  const std::string img_p = arg_of(argc, argv, "--img", "");
  const std::string model = arg_of(argc, argv, "--model", "");
  if (img_p.empty() || model.empty()) {
    printf("usage: latexocr infer --img x.png --model models/m.pt\n");
    return 1;
  }
  const mdl::Cfg c = cfg_of(argc, argv);
  mdl::Params p;
  mdl::build(p, c);
  std::string why;
  if (!mdl::load(p, model, &why)) { printf("%s: %s\n", model.c_str(), why.c_str()); return 1; }
  std::vector<float> img;
  if (!load_gray(img_p, c.H, c.W, img)) { printf("%s が読めません\n", img_p.c_str()); return 1; }
  const std::vector<int> ids = mdl::greedy(img, p, c);
  printf("%s\n", tok::decode(ids).c_str());
  return 0;
}

// データ全部を読ませて、完全一致した割合を出す
static int cmd_eval(int argc, char** argv) {
  const std::string data = arg_of(argc, argv, "--data", "");
  const std::string model = arg_of(argc, argv, "--model", "");
  const int limit = std::atoi(arg_of(argc, argv, "--limit", "100").c_str());
  const bool show = has_flag(argc, argv, "--show-fail");
  if (data.empty() || model.empty()) {
    printf("usage: latexocr eval --data data/val --model models/m.pt [--limit 100]\n");
    return 1;
  }
  const mdl::Cfg c = cfg_of(argc, argv);
  mdl::Params p;
  mdl::build(p, c);
  std::string why;
  if (!mdl::load(p, model, &why)) { printf("%s: %s\n", model.c_str(), why.c_str()); return 1; }
  std::vector<Item> items = read_labels(data);
  if (limit > 0 && (int)items.size() > limit) items.resize((size_t)limit);
  int ok = 0, n = 0, shown = 0;
  for (const Item& it : items) {
    std::vector<float> img;
    if (!load_gray(it.img, c.H, c.W, img)) continue;
    ++n;
    const std::string got = tok::decode(mdl::greedy(img, p, c));
    const std::string want = tok::decode(tok::encode(it.latex));
    if (got == want) ++ok;
    else if (show && shown++ < 10) printf("  NG  正解 %-28s 読み %s\n", want.c_str(), got.c_str());
  }
  printf("完全一致: %d / %d（%.1f%%）\n", ok, n, n ? 100.0 * ok / n : 0.0);
  return 0;
}

// 本家 LaTeX-OCR（pix2tex）の学習済みモデルで読む。
//
// この repo で学習するモデル（pure/model.hpp）とは別物で、こちらは**重みを借りるだけ**。
// 25.5M パラメータ・語彙 1175 の byte-level BPE で、実物の論文の式が読める。
// 手順も本家に合わせてある（前処理 -> image_resizer で大きさを決める -> 貪欲に読む）。
static int cmd_pix2tex(int argc, char** argv) {
  const std::string img_p = arg_of(argc, argv, "--img", "");
  const std::string wp = arg_of(argc, argv, "--weights", "models/pix2tex/weights.pth");
  const std::string rp = arg_of(argc, argv, "--resizer", "models/pix2tex/image_resizer.pth");
  const std::string tp = arg_of(argc, argv, "--tokenizer", "models/pix2tex/tokenizer.json");
  const int max_new = std::atoi(arg_of(argc, argv, "--max-new", "256").c_str());
  const bool no_resize = has_flag(argc, argv, "--no-resize");
  const bool raw = has_flag(argc, argv, "--raw");
  if (img_p.empty()) {
    printf("usage: latexocr pix2tex --img formula.png [--weights models/pix2tex/weights.pth]\n"
           "                        [--resizer ...] [--tokenizer ...] [--no-resize] [--raw]\n");
    return 1;
  }
  ip::Img src;
  if (!load_img(img_p, src)) { printf("%s が読めません\n", img_p.c_str()); return 1; }
  px::Cfg c;
  px::Weights w;
  std::string why;
  if (!w.load(wp, &why)) { printf("%s\n", why.c_str()); return 1; }
  px::Weights rz;
  bool has_rz = false;
  if (!no_resize) {
    FILE* f = fopen(rp.c_str(), "rb");
    if (f) { fclose(f); has_rz = rz.load(rp, &why); }
    if (!has_rz) printf("（%s が無いので大きさは決め打ち。読みは落ちる）\n", rp.c_str());
  }
  bpe::Tokenizer tk;
  if (!tk.load(tp, &why)) { printf("%s\n", why.c_str()); return 1; }
  ip::Img used;
  const std::vector<int> ids = px::read(src, w, c, has_rz ? &rz : nullptr, max_new, &used);
  printf("入力 %dx%d\n", used.w, used.h);
  const std::string latex = tk.decode(ids);
  printf("%s\n", raw ? latex.c_str() : bpe::post_process(latex).c_str());
  return 0;
}

// パリティ用の書き出し。**Python に同じ数を渡す**のが目的なので、乱数で作ったものも
// 計算した結果も全部入れる（重み・画像・入力トークン・正解・ロジット・損失・勾配・貪欲生成）。
// これで違いが出たら、実装の違いである（乱数の引き方でも画像の作り方でもない）。
static int cmd_dump_parity(int argc, char** argv) {
  const std::string out = arg_of(argc, argv, "--out", "");
  const uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "5").c_str(), nullptr, 10);
  if (out.empty()) { printf("usage: latexocr dump-parity --out scratch/parity.bin\n"); return 1; }
  mdl::Cfg c = cfg_of(argc, argv);
  if (!has_flag(argc, argv, "--full")) {              // 既定は小さい設定（速いので）
    c.H = 32; c.W = 96; c.d = 24; c.heads = 3; c.layers = 2; c.ff = 32; c.max_len = 12;
  }
  c.vocab = tok::size();
  mdl::Params p;
  mdl::build(p, c);
  Rng rng(seed);
  mdl::init_params(p, rng);

  std::vector<float> img((size_t)c.H * c.W);
  for (float& v : img) v = (float)rng.unit();          // 中身は何でもよい（両言語で同じなら）
  const std::vector<int> in_ids{tok::BOS, 32, 24, 5, 25};
  const std::vector<int> tgt{32, 24, 5, 25, tok::EOS};
  const int T = (int)in_ids.size();

  std::vector<Tensor> ps = p.all();
  for (Tensor& t : ps) std::fill(t->grad.begin(), t->grad.end(), 0.f);
  const Tensor enc = mdl::encode(mdl::img_tensor(img, c), p, c);
  const Tensor logits = mdl::decode(enc, in_ids, p, c);
  std::vector<float> lg = logits->data;
  const std::vector<float> mask((size_t)tgt.size(), 1.f);
  const Tensor loss = sq::ce_loss(logits, tgt, mask);
  const float lv = loss->data[0];
  backward(loss);
  free_graph(loss);
  const std::vector<int> greedy = mdl::greedy(img, p, c);

  FILE* f = fopen(out.c_str(), "wb");
  if (!f) { printf("%s が書けません\n", out.c_str()); return 1; }
  const auto wi = [&](int32_t v) { fwrite(&v, 4, 1, f); };
  fwrite("LTXPAR01", 1, 8, f);
  wi(c.H); wi(c.W); wi(c.d); wi(c.heads); wi(c.layers); wi(c.ff); wi(c.max_len); wi(c.V()); wi(T);
  fwrite(img.data(), 4, img.size(), f);
  for (int v : in_ids) wi(v);
  for (int v : tgt) wi(v);
  wi((int32_t)p.order.size());
  for (const std::string& n : p.order) {
    wi((int32_t)n.size());
    fwrite(n.data(), 1, n.size(), f);
    const Tensor t = p.t.at(n);
    wi((int32_t)t->shape.size());
    for (int64_t dd : t->shape) wi((int32_t)dd);
    fwrite(t->data.data(), 4, t->data.size(), f);
    fwrite(t->grad.data(), 4, t->grad.size(), f);
  }
  fwrite(lg.data(), 4, lg.size(), f);
  fwrite(&lv, 4, 1, f);
  wi((int32_t)greedy.size());
  for (int v : greedy) wi(v);
  fclose(f);
  printf("wrote %s（重み %zu 本、ロジット %zu、損失 %.6f、貪欲生成 %zu トークン）\n", out.c_str(),
         p.order.size(), lg.size(), lv, greedy.size());
  return 0;
}

// トークナイザの往復と、勾配の検算
static int cmd_selftest(int argc, char** argv) {
  const int n = std::atoi(arg_of(argc, argv, "--n", "500").c_str());
  Rng rng(3);
  int bad = 0, checked = 0;
  for (int i = 0; i < n; ++i) {
    std::string why;
    const ex::E e = ex::parse(gx::one(rng), &why);
    if (!why.empty()) continue;
    const std::string s = ex::to_latex(e);
    const std::vector<int> ids = tok::encode(s);
    bool unk = false;
    for (int id : ids)
      if (id == tok::UNK) unk = true;
    if (unk) continue;
    ++checked;
    if (tok::decode(tok::encode(tok::decode(ids))) != tok::decode(ids)) {
      ++bad;
      if (bad <= 5) printf("  NG  %s -> %s\n", s.c_str(), tok::decode(ids).c_str());
    }
  }
  printf("トークナイザの往復: %d / %d 一致\n", checked - bad, checked);

  // 勾配の検算（小さい設定で、数値微分と突き合わせる）
  mdl::Cfg c;
  c.H = 32;
  c.W = 64;
  c.d = 16;
  c.heads = 2;
  c.layers = 1;
  c.ff = 24;
  c.max_len = 8;
  mdl::Params p;
  mdl::build(p, c);
  Rng r2(9);
  mdl::init_params(p, r2);
  std::vector<float> img((size_t)c.H * c.W);
  for (float& v : img) v = (float)r2.unit();
  const std::vector<int> in{tok::BOS, 5, 6};
  const std::vector<int> tgt{5, 6, tok::EOS};
  const std::vector<float> mask(3, 1.f);
  const auto fwd = [&]() {
    const Tensor enc = mdl::encode(mdl::img_tensor(img, c), p, c);
    const Tensor lg = mdl::decode(enc, in, p, c);
    return sq::ce_loss(lg, tgt, mask);
  };
  std::vector<Tensor> ps = p.all();
  for (Tensor& t : ps) std::fill(t->grad.begin(), t->grad.end(), 0.f);
  Tensor loss = fwd();
  backward(loss);
  const double l0 = loss->data[0];
  free_graph(loss);

  // **方向微分で比べる**（1 点ずつだと float32 のノイズに埋まる。上の説明を参照）。
  // ランダムな向き v を引いて、<g, v> と (L(θ+hv) - L(θ-hv)) / 2h を突き合わせる。
  double worst = 0, wa = 0, wn = 0;
  const int ndir = 5;
  for (int dir = 0; dir < ndir; ++dir) {
    std::vector<std::vector<float>> v(ps.size());
    double dot = 0;
    for (size_t k = 0; k < ps.size(); ++k) {
      v[k].resize((size_t)ps[k]->numel());
      for (size_t i = 0; i < v[k].size(); ++i) {
        v[k][i] = (float)(r2.unit() * 2.0 - 1.0);
        dot += (double)v[k][i] * ps[k]->grad[i];
      }
    }
    const float h = 1e-3f;
    const auto shift = [&](float sgn) {
      for (size_t k = 0; k < ps.size(); ++k)
        for (size_t i = 0; i < v[k].size(); ++i) ps[k]->data[i] += sgn * h * v[k][i];
    };
    shift(+1.f);
    Tensor l1 = fwd();
    const double a1 = l1->data[0];
    free_graph(l1);
    shift(-2.f);
    Tensor l2 = fwd();
    const double a2 = l2->data[0];
    free_graph(l2);
    shift(+1.f);                                     // 元に戻す
    const double num = (a1 - a2) / (2 * h);
    const double rel = std::fabs(dot - num) / std::max(1e-3, std::max(std::fabs(dot), std::fabs(num)));
    if (rel > worst) { worst = rel; wa = dot; wn = num; }
  }
  printf("勾配の検算（方向微分 %d 本）: loss %.6f、最悪の相対誤差 %.3e（解析 %.4f / 数値 %.4f）\n",
         ndir, l0, worst, wa, wn);
  const bool pass = bad == 0 && worst < 2e-2;
  printf("selftest: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "vocab") return cmd_vocab();
  if (cmd == "tok") return cmd_tok(argc, argv);
  if (cmd == "gen") return cmd_gen(argc, argv);
  if (cmd == "init") return cmd_init(argc, argv);
  if (cmd == "train") return cmd_train(argc, argv);
  if (cmd == "infer") return cmd_infer(argc, argv);
  if (cmd == "eval") return cmd_eval(argc, argv);
  if (cmd == "selftest") return cmd_selftest(argc, argv);
  if (cmd == "dump-parity") return cmd_dump_parity(argc, argv);
  if (cmd == "pix2tex") return cmd_pix2tex(argc, argv);
  printf("usage: latexocr <vocab|tok|gen|init|train|infer|eval|selftest|dump-parity|pix2tex> ...\n");
  return 1;
}
