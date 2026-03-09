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

function formatField(value: string | null | undefined) {
  return value ?? '';
}

export function renderPostsMarkdown(posts: RednotePost[]) {
  if (posts.length === 0) {
    return '没有获取到帖子。\n';
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
