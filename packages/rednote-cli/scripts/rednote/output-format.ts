import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RednotePost } from './post-types.ts';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REDNOTE_ROOT = path.resolve(SCRIPT_DIR, '../..');

export type OutputFormat = 'json' | 'md';

export type OutputCliValues = {
  instance?: string;
  keyword?: string;
  format: OutputFormat;
  saveRequested: boolean;
  savePath?: string;
  help?: boolean;
};

function parseOptionWithEquals(arg: string) {
  const equalIndex = arg.indexOf('=');
  if (equalIndex === -1) {
    return null;
  }

  return {
    key: arg.slice(0, equalIndex),
    value: arg.slice(equalIndex + 1),
  };
}

export function parseOutputCliArgs(argv: string[], options: { includeKeyword?: boolean } = {}): OutputCliValues {
  const values: OutputCliValues = {
    format: 'md',
    saveRequested: false,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const withEquals = parseOptionWithEquals(arg);

    if (arg === '-h' || arg === '--help') {
      values.help = true;
      continue;
    }

    if (withEquals?.key === '--instance') {
      values.instance = withEquals.value;
      continue;
    }

    if (arg === '--instance') {
      values.instance = argv[index + 1];
      index += 1;
      continue;
    }

    if (options.includeKeyword && withEquals?.key === '--keyword') {
      values.keyword = withEquals.value;
      continue;
    }

    if (options.includeKeyword && arg === '--keyword') {
      values.keyword = argv[index + 1];
      index += 1;
      continue;
    }

    if (withEquals?.key === '--format') {
      const format = withEquals.value;
      if (format !== 'json' && format !== 'md') {
        throw new Error(`Invalid --format value: ${format}`);
      }
      values.format = format;
      continue;
    }

    if (arg === '--format') {
      const format = argv[index + 1];
      if (format !== 'json' && format !== 'md') {
        throw new Error(`Invalid --format value: ${String(format)}`);
      }
      values.format = format;
      index += 1;
      continue;
    }

    if (withEquals?.key === '--save') {
      values.saveRequested = true;
      values.savePath = withEquals.value;
      continue;
    }

    if (arg === '--save') {
      values.saveRequested = true;
      const nextArg = argv[index + 1];
      if (nextArg && !nextArg.startsWith('-')) {
        values.savePath = nextArg;
        index += 1;
      }
      continue;
    }
  }

  return values;
}

function slugifyKeyword(keyword: string) {
  return keyword
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\p{Letter}\p{Number}-]+/gu, '')
    .slice(0, 32) || 'query';
}

function timestampForFilename() {
  return new Date().toISOString().replaceAll(':', '').replaceAll('.', '').replace('T', '-').replace('Z', 'Z');
}

export function resolveSavePath(command: 'home' | 'search', explicitPath?: string, keyword?: string) {
  if (explicitPath) {
    return path.resolve(explicitPath);
  }

  const keywordSuffix = keyword ? `-${slugifyKeyword(keyword)}` : '';
  return path.join(REDNOTE_ROOT, 'output', `${command}${keywordSuffix}-${timestampForFilename()}.jsonl`);
}

export function writePostsJsonl(posts: RednotePost[], filePath: string) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const content = posts.map((post) => JSON.stringify(post)).join('\n');
  fs.writeFileSync(filePath, content ? `${content}\n` : '', 'utf8');
}

export function ensureJsonSavePath(format: OutputFormat, savePath?: string) {
  if (format !== 'json') {
    return;
  }

  if (!savePath?.trim()) {
    throw new Error('The --save PATH option is required when --format json is used.');
  }
}

export function resolveJsonSavePath(explicitPath?: string) {
  const normalizedPath = explicitPath?.trim();
  if (!normalizedPath) {
    throw new Error('The --save PATH option is required when --format json is used.');
  }

  return path.resolve(normalizedPath);
}

export function writeJsonFile(payload: unknown, filePath: string) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

type JsonFieldExample = string | JsonFieldExample[] | { [key: string]: JsonFieldExample };

function describeStringValue(value: string, key: string) {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value) || key.toLowerCase().endsWith('at')) {
    return 'ISO-8601 string';
  }

  if (key.toLowerCase().endsWith('url')) {
    return 'string (URL)';
  }

  if (key.toLowerCase().endsWith('path')) {
    return 'string (path)';
  }

  return 'string';
}

function buildJsonFieldExample(value: unknown, key = '', depth = 0): JsonFieldExample {
  if (value === null) {
    return 'null';
  }

  if (value === undefined) {
    return 'undefined';
  }

  if (typeof value === 'string') {
    return describeStringValue(value, key);
  }

  if (typeof value === 'number') {
    return 'number';
  }

  if (typeof value === 'boolean') {
    return 'boolean';
  }

  if (Array.isArray(value)) {
    return value.length > 0 ? [buildJsonFieldExample(value[0], key, depth + 1)] : ['unknown'];
  }

  if (typeof value === 'object') {
    if (depth >= 3 || key === 'raw') {
      return 'object';
    }

    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return 'object';
    }

    return Object.fromEntries(entries.map(([entryKey, entryValue]) => [entryKey, buildJsonFieldExample(entryValue, entryKey, depth + 1)]));
  }

  return typeof value;
}

export function renderJsonSaveSummary(filePath: string, payload: unknown) {
  return `Saved JSON: ${filePath}\n\nField format example:\n${JSON.stringify(buildJsonFieldExample(payload), null, 3)}\n`;
}

function formatField(value: string | null | undefined) {
  return value ?? '';
}

export function renderPostsMarkdown(posts: RednotePost[]) {
  if (posts.length === 0) {
    return 'No posts were captured.\n';
  }

  return `${posts
    .map((post) => [
      `- id: ${post.id}`,
      `- displayTitle: ${formatField(post.noteCard.displayTitle)}`,
      `- likedCount: ${formatField(post.noteCard.interactInfo.likedCount)}`,
      `- url: ${post.url}`,
      `- nickName: ${formatField(post.noteCard.user.nickName)}`,
      `- userId: ${formatField(post.noteCard.user.userId)}`,
    ].join('\n'))
    .join('\n\n')}\n`;
}
