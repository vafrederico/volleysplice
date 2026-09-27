#!/usr/bin/env node
// Replay independent Python-generated contract fixtures without a browser or ORT.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, basename } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { decodeRallies, fuseFeatures, roundTokensToFloat16 } from './browser-distilled-large-contract.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
if (!args.has('--fixtures') || !args.has('--output')) throw new Error('--fixtures and --output are required');
const root = resolve(args.get('--fixtures'));
const bytes = await readFile(join(root, 'manifest.json'));
const manifest = JSON.parse(bytes), contract = manifest.temporalContract;
if (!contract) throw new Error('Independent temporal fixtures are missing');
const read = async entry => {
  if (entry.file !== basename(entry.file)) throw new Error('Fixture must be local');
  const data = await readFile(join(root, entry.file));
  if (data.byteLength !== entry.sizeBytes || createHash('sha256').update(data).digest('hex') !== entry.sha256)
    throw new Error('Fixture hash mismatch');
  const buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return entry.dtype === 'float64' ? new Float64Array(buffer) : new Float32Array(buffer);
};
const compare = (actual, expected, name) => {
  if (actual.length !== expected.length) throw new Error(`${name}: dimensions differ`);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(error) || error !== 0) throw new Error(`${name}: expected exact float32 at ${i}, error ${error}`);
    maximum = Math.max(maximum, error);
  }
  return { passed: true, maximumAbsoluteError: maximum, values: actual.length };
};
const decoder = contract.decoder.map(test => {
  const actual = decodeRallies(new Float64Array(test.times), new Float32Array(test.probabilities), test.duration, test.config)
    .map(({ start, end }) => ({ start, end }));
  if (!isDeepStrictEqual(actual, test.expected)) throw new Error(`${test.id}: decoded boundaries differ`);
  return { id: test.id, passed: true, rallies: actual.length };
});
const fixture = contract.fusion, input = {};
for (const [name, entry] of Object.entries(fixture.inputs)) input[name] = await read(entry);
const tokenRounding = compare(roundTokensToFloat16(input.rawTokens), input.tokens, 'tokens');
const fusion = [];
for (const test of fixture.variants) {
  const actual = fuseFeatures(input.times, input.rankedAv, input.embeddingTimes, input.tokens, input.quality, test.config);
  fusion.push({ id: test.id, ...compare(actual, await read(test.expected), test.id) });
}
const report = { schemaVersion: 1, passed: true, fixtureManifestSha256: createHash('sha256').update(bytes).digest('hex'),
  decoder, tokenRounding, fusion };
await writeFile(resolve(args.get('--output')), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ passed: true, decoderCases: decoder.length, fusionCases: fusion.length, tokenValues: input.tokens.length }));
