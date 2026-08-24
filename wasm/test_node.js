// ブラウザに出す前に、wasm を node で動かして確かめる。
//
// **確かめるのは「同じ答えが出るか」**。docs/ の wasm が読んだ LaTeX と、
// latexocr.exe infer が読んだ LaTeX が 1 文字も違わないこと。ここが合っていれば、
// デモページで見えているものは native と同じ推論である。
//
//   node wasm/test_node.js [枚数]
//
// 手順: lx_sample ででたらめな式を作り（正解つき）、それを png に書き出して
// latexocr.exe にも読ませ、両者を突き合わせる。
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const N = parseInt(process.argv[2] || '20', 10);
const createLatexOCR = require(path.join(ROOT, 'docs', 'latexocr.js'));

// 最小の png 書き出し（学習にも推論にも使わないので、ここだけで完結させる）
function writeGrayPng(file, w, h, gray) {
  const zlib = require('zlib');
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; ++y) {
    raw[y * (w + 1)] = 0;                                  // filter type 0
    for (let x = 0; x < w; ++x) raw[y * (w + 1) + 1 + x] = gray[y * w + x];
  }
  const crcTable = [];
  for (let n = 0; n < 256; ++n) {
    let c = n;
    for (let k = 0; k < 8; ++k) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const c = Buffer.alloc(4);
    c.writeUInt32BE(crc(body));
    return Buffer.concat([len, body, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 0;                                // 8bit grayscale
  fs.writeFileSync(file, Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}

createLatexOCR().then((M) => {
  const rc = M.ccall('lx_init', 'number', ['string', 'string'], ['fonts/math.ttf', 'models/model.pt']);
  if (rc !== 0) {
    console.error('lx_init 失敗:', M.ccall('lx_why', 'string', [], []));
    process.exit(1);
  }
  const W = M.ccall('lx_w', 'number', [], []);
  const H = M.ccall('lx_h', 'number', [], []);
  const buf = M._malloc(W * H);
  const exe = path.join(ROOT, process.platform === 'win32' ? 'latexocr.exe' : 'latexocr.exe');
  const model = path.join(ROOT, 'models', 'wasm_check.pt');
  const tmp = path.join(ROOT, 'scratch', 'wasm_check.png');
  fs.mkdirSync(path.join(ROOT, 'scratch'), { recursive: true });

  let okTruth = 0, okNative = 0, n = 0, nativeRan = fs.existsSync(exe) && fs.existsSync(model);
  for (let i = 0; i < N; ++i) {
    if (!M.ccall('lx_sample', 'number', ['number', 'number'], [1000 + i, buf])) continue;
    const truth = M.ccall('lx_why', 'string', [], []);
    const gray = M.HEAPU8.slice(buf, buf + W * H);
    const got = M.ccall('lx_read', 'string', ['number'], [buf]);
    ++n;
    if (got === truth) ++okTruth;
    if (nativeRan) {
      writeGrayPng(tmp, W, H, gray);
      const out = execFileSync(exe, ['infer', '--img', tmp, '--model', model],
                               { cwd: ROOT }).toString().trim();
      if (out === got) ++okNative;
      else if (okNative + 3 > n) console.log('  ちがう\n    wasm   %s\n    native %s', got, out);
    }
  }
  console.log('%d 枚: wasm が正解と一致 %d (%.0f%%)', n, okTruth, 100 * okTruth / n);
  if (nativeRan) {
    console.log('       wasm と native が一致 %d/%d  %s', okNative, n, okNative === n ? 'ok' : 'NG');
    process.exit(okNative === n ? 0 : 1);
  } else {
    console.log('       native との突き合わせは飛ばした（%s か %s が無い）', exe, model);
  }
  M._free(buf);
});
