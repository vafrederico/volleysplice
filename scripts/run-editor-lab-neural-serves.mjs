import { createServer, request } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve, basename } from "node:path";
const args = new Map(process.argv.slice(2).reduce((pairs, value, i, values) => i % 2 ? pairs : [...pairs, [value, values[i + 1]]], []));
const require = createRequire(resolve(args.get("--harness"), "package.json"));
const { chromium } = require("playwright-core");
const work = resolve("artifacts/private-media/editor-lab-neural-serves");
const server = createServer(async (req, res) => {
  if (req.url.startsWith("/lab/")) {
    const proxy = request({ hostname: "127.0.0.1", port: 3000, path: req.url.slice(4), method: req.method, headers: req.headers }, upstream => {
      res.writeHead(upstream.statusCode, upstream.headers); upstream.pipe(res);
    });
    proxy.on("error", error => { res.writeHead(502); res.end(String(error)); });
    req.pipe(proxy); return;
  }
  try {
    if (req.url === "/") { res.setHeader("Content-Type", "text/html"); res.end('<!doctype html><title>Neural serve inference</title><script type="module" src="/runner.js"></script>'); return; }
    const name = basename(req.url.split("?")[0]);
    const path = req.url.startsWith("/runtime/") ? resolve("prod/public/runtime", name) : resolve(work, name);
    if (!req.url.startsWith("/runtime/") && !["runner.js", "input.json"].includes(name)) { res.writeHead(404); res.end(); return; }
    res.setHeader("Content-Type", name.endsWith(".json") ? "application/json" : "application/javascript");
    res.end(await readFile(path));
  } catch (error) { res.writeHead(500); res.end(String(error)); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", headless: true });
try {
  const page = await browser.newPage();
  page.on("console", message => console.log(message.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => window.neuralServeResult || window.neuralServeError, null, { timeout: 3600000 });
  const result = await page.evaluate(() => ({ result: window.neuralServeResult, error: window.neuralServeError }));
  if (result.error) throw Error(result.error);
  await writeFile(resolve(work, "results.json"), JSON.stringify(result.result));
  console.log(JSON.stringify({ passed: true, counts: result.result.counts }));
} finally { await browser.close(); server.close(); }
