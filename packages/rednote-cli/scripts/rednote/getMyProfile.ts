#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import { runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn } from './checkLogin.ts';
import { getProfile, type ProfileFormat, type ProfileMode, type RednoteProfileResult, selectProfileOutput, renderProfileMarkdown } from './getProfile.ts';
import { ensureJsonSavePath, renderJsonSaveSummary, resolveJsonSavePath, writeJsonFile } from './output-format.ts';
import { persistProfile } from './persistence.ts';

export type GetMyProfileCliValues = {
  instance?: string;
  format: ProfileFormat;
  mode: ProfileMode;
  maxNotes: number;
  savePath?: string;
  help?: boolean;
};

function printGetMyProfileHelp() {
  process.stdout.write(`rednote get-my-profile

Usage:
  npx -y @skills-store/rednote get-my-profile [--instance NAME] [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]
  node --experimental-strip-types ./scripts/rednote/getMyProfile.ts --instance NAME [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]
  bun ./scripts/rednote/getMyProfile.ts --instance NAME [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --mode MODE       Optional. profile | notes. Default: profile
  --max-notes N     Optional. Max notes to fetch by scrolling. Default: 100
  --format FORMAT   Output format: md | json. Default: md
  --save PATH       Required when --format json is used. Saves only the selected mode data as JSON
  -h, --help        Show this help
`);
}

export function parseGetMyProfileCliArgs(argv: string[]): GetMyProfileCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      format: { type: 'string' },
      mode: { type: 'string' },
      'max-notes': { type: 'string' },
      save: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  const format = values.format ?? 'md';
  if (format !== 'md' && format !== 'json') {
    throw new Error(`Invalid --format value: ${String(format)}`);
  }

  const mode = values.mode ?? 'profile';
  if (mode !== 'profile' && mode !== 'notes') {
    throw new Error(`Invalid --mode value: ${String(values.mode)}`);
  }

  const maxNotesValue = values['max-notes'] ?? '100';
  const maxNotes = parseInt(maxNotesValue, 10);
  if (isNaN(maxNotes) || maxNotes < 1) {
    throw new Error(`Invalid --max-notes value: ${maxNotesValue}`);
  }

  return {
    instance: values.instance,
    format,
    mode,
    maxNotes,
    savePath: values.save,
    help: values.help,
  };
}

function writeMyProfileOutput(result: RednoteProfileResult, values: GetMyProfileCliValues) {
  const output = selectProfileOutput(result, values.mode);

  if (values.format === 'json') {
    const savedPath = resolveJsonSavePath(values.savePath);
    writeJsonFile(output, savedPath);
    process.stdout.write(renderJsonSaveSummary(savedPath, output));
    return;
  }

  process.stdout.write(renderProfileMarkdown(result, values.mode));
}

async function navigateToMyProfile(page: import('playwright-core').Page): Promise<{ userId: string; profileUrl: string }> {
  // Navigate to explore page
  await page.goto('https://www.xiaohongshu.com/explore', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(2000);

  // Click user profile link
  const userLink = page.locator('.user.side-bar-component').first();
  await userLink.click();

  // Wait for navigation to profile page
  await page.waitForURL(/\/user\/profile\//, { timeout: 10000 });
  await page.waitForTimeout(1000);

  // Extract userId from URL
  const profileUrl = page.url();
  const match = profileUrl.match(/\/user\/profile\/([^/?]+)/);
  if (!match) {
    throw new Error(`Failed to extract userId from profile URL: ${profileUrl}`);
  }

  const userId = match[1];
  return { userId, profileUrl };
}

export async function runGetMyProfileCommand(values: GetMyProfileCliValues = { format: 'md', mode: 'profile', maxNotes: 100 }) {
  if (values.help) {
    printGetMyProfileHelp();
    return;
  }

  ensureJsonSavePath(values.format, values.savePath);

  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'fetching my profile', session);

    // Navigate to profile page by clicking user link
    const { userId, profileUrl } = await navigateToMyProfile(session.page);

    const result = await getProfile(session, profileUrl, userId, values.maxNotes);

    // Persist profile and notes to database
    await persistProfile({
      instanceName: target.instanceName,
      result,
    });

    writeMyProfileOutput(result, values);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseGetMyProfileCliArgs(process.argv.slice(2));
  await runGetMyProfileCommand(values);
}

runCli(import.meta.url, main);
