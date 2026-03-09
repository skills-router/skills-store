#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { getRednoteEnvironmentInfo } from '../utils/browser-core.ts';

export type EnvCliValues = {
  format?: 'md' | 'json';
  help?: boolean;
};

function printEnvHelp() {
  process.stdout.write(`rednote env

Usage:
  npx -y @skills-store/rednote env [--format md|json]
  node --experimental-strip-types ./scripts/rednote/env.ts [--format md|json]
  bun ./scripts/rednote/env.ts [--format md|json]

Options:
  --format FORMAT   Output format: md | json. Default: md
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
  if (format === 'json') {
    printJson(getRednoteEnvironmentInfo());
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
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.format && values.format !== 'md' && values.format !== 'json') {
    throw new Error(`Invalid --format value: ${String(values.format)}`);
  }

  await runEnvCommand(values as EnvCliValues);
}

runCli(import.meta.url, main);
