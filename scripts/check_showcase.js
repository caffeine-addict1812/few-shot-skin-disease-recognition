// Check that the showcase page reproduces the report: score all 600 fixed test episodes with the page's own
// maths (showcase/proto.js) on the page's own data (showcase/data.js), and compare with results/*_test.json.
//   node scripts/check_showcase.js
const fs = require("fs"), path = require("path"), vm = require("vm");
const dir = path.join(__dirname, "..", "site", "assets");
const Proto = require(path.join(dir, "proto.js"));
const ctx = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(dir, "data.js"), "utf8"), ctx);
const S = ctx.window.SHOWCASE;
let worst = 0;
for (const model of ["finetuned", "frozen"]) {
  const Z = Proto.decodeF16(S.emb[model]);
  for (const k of [1, 5]) {
    const key = `5way_${k}shot`;
    const eps = S.episodes[key];
    const acc = 100 * eps.reduce((a, ep) => a + Proto.episodeAccuracy(Z, S.dim, ep, 5, k, 5, S.scale[model]), 0) / eps.length;
    const ref = S.stored_acc[model][key];
    worst = Math.max(worst, Math.abs(acc - ref));
    console.log(`${model.padEnd(9)} ${key}: page ${acc.toFixed(2)}  report ${ref.toFixed(2)}  diff ${(acc - ref).toFixed(2)}`);
  }
}
if (worst > 0.5) { console.error(`FAIL: page differs from the report by ${worst.toFixed(2)} pp`); process.exit(1); }
console.log(`OK: page matches the report within ${worst.toFixed(2)} pp (fp16 embeddings, CPU fp32 vs GPU bf16)`);
