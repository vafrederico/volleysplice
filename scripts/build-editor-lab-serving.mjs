import { createRequire } from 'node:module';
import { resolve } from 'node:path';
const require = createRequire(resolve('prod/package.json'));
const { build } = require('esbuild');
if (!process.argv[2]) throw Error('Specify external output bundle path');
await build({ entryPoints: ['scripts/editor-lab-serving-browser.ts'], bundle: true,
  format: 'esm', platform: 'browser', define: { 'import.meta.env.BASE_URL': JSON.stringify('/') },
  outfile: process.argv[2] });
