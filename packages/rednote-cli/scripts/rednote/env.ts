#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import { runCli } from '../utils/browser-cli.ts';
import { getRednoteEnvironmentInfo } from '../utils/browser-core.ts';
import { ensureJsonSavePath, renderJsonSaveSummary, resolveJsonSavePath, writeJsonFile } from './output-format.ts';

export type EnvCliValues = {
  format?: 'md' | 'json';
  savePath?: string;
  help?: boolean;
};

function printEnvHelp() {
  process.stdout.write(`rednote env

Usage:
  npx -y @skills-store/rednote env [--format md|json] [--save PATH]
  node --experimental-strip-types ./scripts/rednote/env.ts [--format md|json] [--save PATH]
  bun ./scripts/rednote/env.ts [--format md|json] [--save PATH]

Options:
  --format FORMAT   Output format: md | json. Default: md
  --save PATH       Required when --format json is used. Saves the full result as JSON
  -h, --help        Show this help
`);
}

function renderEnvironmentMarkdown() {
  const info = getRednoteEnvironmentInfo();
  return [
    '## Environment',
    '',
    `- Platform: ${info.platform}`,
    `- Node: ${info.nodeVersion}`,
    `- Home: ${info.homeDir}`,
    `- Package Root: ${info.packageRoot}`,
    `- Storage Home: ${info.storageHome}`,
    `- Storage Root: ${info.storageRoot}`,
    `- Instances Dir: ${info.instancesDir}`,
    `- Instance Store: ${info.instanceStorePath}`,
    `- Legacy Package Instances: ${info.legacyPackageInstancesDir}`,
    '',
    'Custom browser instances and metadata are stored under `~/.skills-router/rednote/instances`.',
    '',
  ].join('\n');
}

export async function runEnvCommand(values: EnvCliValues = {}) {
  if (values.help) {
    printEnvHelp();
    return;
  }

  const format = values.format ?? 'md';
  ensureJsonSavePath(format, values.savePath);

  if (format === 'json') {
    const result = getRednoteEnvironmentInfo();
    const savedPath = resolveJsonSavePath(values.savePath);
    writeJsonFile(result, savedPath);
    process.stdout.write(renderJsonSaveSummary(savedPath, result));
    return;
  }

  process.stdout.write(renderEnvironmentMarkdown());
}

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    strict: false,
    options: {
      format: { type: 'string' },
      save: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.format && values.format !== 'md' && values.format !== 'json') {
    throw new Error(`Invalid --format value: ${String(values.format)}`);
  }

  await runEnvCommand({
    format: values.format as 'md' | 'json' | undefined,
    savePath: values.save,
    help: values.help,
  });
}

runCli(import.meta.url, main);
