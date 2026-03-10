#!/usr/bin/env -S node --experimental-strip-types

import * as cheerio from 'cheerio';
import { parseArgs } from 'node:util';
import vm from 'node:vm';
import type { Page, Response } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn, type RednoteSession } from './checkLogin.ts';

export type FeedDetailFormat = 'json' | 'md';

export type FeedDetailCliValues = {
  instance?: string;
  urls: string[];
  format: FeedDetailFormat;
  help?: boolean;
};

export type RednoteComment = {
  id: string | null;
  content: string | null;
  userId: string | null;
  nickname: string | null;
  likedCount: string | null;
  subCommentCount: number | null;
  raw: unknown;
};

export type RednoteDetailNote = {
  noteId: string | null;
  title: string | null;
  desc: string | null;
  type: string | null;
  interactInfo: {
    liked: boolean | null;
    likedCount: string | null;
    commentCount: string | null;
    collectedCount: string | null;
    shareCount: string | null;
    collected: boolean | null;
    followed: boolean | null;
  };
  tagList: Array<{
    name: string | null;
  }>;
  imageList: Array<{
    urlDefault: string | null;
    urlPre: string | null;
    width: number | null;
    height: number | null;
  }>;
  video: {
    url: string | null;
    raw: unknown;
  } | null;
  raw: unknown;
};

export type RednoteFeedDetailItem = {
  url: string;
  note: RednoteDetailNote;
  comments: RednoteComment[];
};

export type FeedDetailResult = {
  ok: true;
  detail: {
    fetchedAt: string;
    total: number;
    items: RednoteFeedDetailItem[];
  };
};

function printGetFeedDetailHelp() {
  process.stdout.write(`rednote get-feed-detail

Usage:
  npx -y @skills-store/rednote get-feed-detail [--instance NAME] --url URL [--url URL] [--format md|json]
  node --experimental-strip-types ./scripts/rednote/getFeedDetail.ts --instance NAME --url URL [--url URL] [--format md|json]
  bun ./scripts/rednote/getFeedDetail.ts --instance NAME --url URL [--url URL] [--format md|json]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --url URL         Required. Xiaohongshu explore url, repeatable
  --format FORMAT   Output format: md | json. Default: md
  -h, --help        Show this help
`);
}

export function parseGetFeedDetailCliArgs(argv: string[]): FeedDetailCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      url: { type: 'string', multiple: true },
      format: { type: 'string' },
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

  return {
    instance: values.instance,
    urls: values.url ?? [],
    format,
    help: values.help,
  };
}

function validateFeedDetailUrl(url: string) {
  try {
    const parsed = new URL(url);
    if (!parsed.href.startsWith('https://www.xiaohongshu.com/explore/')) {
      throw new Error(`url is not valid: ${url},must start with "https://www.xiaohongshu.com/explore/"`);
    }
    if (!parsed.searchParams.get('xsec_token')) {
      throw new Error(`url is not valid: ${url},must include "xsec_token="`);
    }
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`url is not valid: ${url}`);
    }
    throw error;
  }
}

function normalizeFeedDetailUrl(url: string) {
  try {
    const parsed = new URL(url);
    if (!parsed.searchParams.has('xsec_source')) {
      parsed.searchParams.set('xsec_source', 'pc_feed');
    }
    return parsed.toString();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`url is not valid: ${url}`);
    }
    throw error;
  }
}

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  return session.page;
}

function extractVideoUrl(note: any) {
  const streams = Object.values(note?.video?.media?.stream ?? {}) as any[];
  const firstAvailable = streams.find((items) => Array.isArray(items) && items.length > 0);
  return firstAvailable?.[0]?.backupUrls?.[0] ?? null;
}

function normalizeDetailNote(note: any): RednoteDetailNote {
  return {
    noteId: note?.noteId ?? null,
    title: note?.title ?? null,
    desc: note?.desc ?? null,
    type: note?.type ?? null,
    interactInfo: {
      liked: note?.interactInfo?.liked ?? null,
      likedCount: note?.interactInfo?.likedCount ?? null,
      commentCount: note?.interactInfo?.commentCount ?? null,
      collected: note?.interactInfo?.collected ?? null,
      collectedCount: note?.interactInfo?.collectedCount ?? null,
      shareCount: note?.interactInfo?.shareCount ?? null,
      followed: note?.interactInfo?.followed ?? null,
    },
    tagList: Array.isArray(note?.tagList)
      ? note.tagList.map((tag: any) => ({ name: tag?.name ?? null }))
      : [],
    imageList: Array.isArray(note?.imageList)
      ? note.imageList.map((image: any) => ({
          urlDefault: image?.urlDefault ?? null,
          urlPre: image?.urlPre ?? null,
          width: image?.width ?? null,
          height: image?.height ?? null,
        }))
      : [],
    video: note?.video
      ? {
          url: extractVideoUrl(note),
          raw: note.video,
        }
      : null,
    raw: note,
  };
}

function normalizeComments(comments: any[]): RednoteComment[] {
  return comments.map((comment) => ({
    id: comment?.id ?? comment?.commentId ?? null,
    content: comment?.content ?? null,
    userId: comment?.userInfo?.userId ?? null,
    nickname: comment?.userInfo?.nickname ?? null,
    likedCount: comment?.interactInfo?.likedCount ?? null,
    subCommentCount: typeof comment?.subCommentCount === 'number' ? comment.subCommentCount : null,
    raw: comment,
  }));
}

function formatDetailField(value: string | number | boolean | null | undefined) {
  return value ?? '';
}

function renderDetailMarkdown(items: RednoteFeedDetailItem[]) {
  if (items.length === 0) {
    return '没有获取到帖子详情。\n';
  }

  return `${items.map((item) => {
    const lines: string[] = [];

    lines.push('## Note');
    lines.push('');
    lines.push(`- Url: ${item.url}`);
    lines.push(`- Title: ${formatDetailField(item.note.title)}`);
    lines.push(`- Type: ${formatDetailField(item.note.type)}`);
    lines.push(`- Liked: ${formatDetailField(item.note.interactInfo.liked)}`);
    lines.push(`- Collected: ${formatDetailField(item.note.interactInfo.collected)}`);
    lines.push(`- LikedCount: ${formatDetailField(item.note.interactInfo.likedCount)}`);
    lines.push(`- CommentCount: ${formatDetailField(item.note.interactInfo.commentCount)}`);
    lines.push(`- CollectedCount: ${formatDetailField(item.note.interactInfo.collectedCount)}`);
    lines.push(`- ShareCount: ${formatDetailField(item.note.interactInfo.shareCount)}`);
    lines.push(`- Tags: ${item.note.tagList.map((tag) => tag.name ? `#${tag.name}` : '').filter(Boolean).join(' ')}`);
    lines.push('');
    lines.push('## Content');
    lines.push('');
    lines.push(item.note.desc ?? '');

    if (item.note.imageList.length > 0 || item.note.video?.url) {
      lines.push('');
      lines.push('## Media');
      lines.push('');

      item.note.imageList.forEach((image, index) => {
        if (image.urlDefault) {
          lines.push(`- Image${index + 1}: ${image.urlDefault}`);
        }
      });

      if (item.note.video?.url) {
        lines.push(`- Video: ${item.note.video.url}`);
      }
    }

    lines.push('');
    lines.push('## Comments');
    lines.push('');

    if (item.comments.length === 0) {
      lines.push('- Comments not found');
    } else {
      item.comments.forEach((comment) => {
        const prefix = comment.nickname ? `${comment.nickname}: ` : '';
        lines.push(`- ${prefix}${comment.content ?? ''}`);
      });
    }

    return lines.join('\n');
  }).join('\n\n---\n\n')}\n`;
}

async function captureFeedDetail(page: Page, targetUrl: string): Promise<RednoteFeedDetailItem> {
  let note: any = null;
  let comments: any[] | null = null;

  const handleResponse = async (response: Response) => {
    try {
      const url = new URL(response.url());
      if (response.status() !== 200) {
        return;
      }

      if (url.href.includes('/explore/')) {
        const html = await response.text();
        const $ = cheerio.load(html);

        $('script').each((_, element) => {
          const scriptContent = $(element).html();
          if (!scriptContent?.includes('window.__INITIAL_STATE__')) {
            return;
          }

          const scriptText = scriptContent.substring(scriptContent.indexOf('=') + 1);
          const sandbox: { info?: any } = {};
          vm.createContext(sandbox);
          vm.runInContext(`var info = ${scriptText}`, sandbox);
          const noteState = sandbox.info?.note;
          if (noteState?.noteDetailMap && noteState?.currentNoteId) {
            note = noteState.noteDetailMap[noteState.currentNoteId]?.note ?? note;
          }
        });
      } else if (url.href.includes('comment/page?')) {
        const data = await response.json() as { data?: { comments?: any[] } };
        comments = Array.isArray(data?.data?.comments) ? data.data.comments : [];
      }
    } catch {
    }
  };

  page.on('response', handleResponse);
  try {
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });

    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      if (note && comments !== null) {
        break;
      }
      await page.waitForTimeout(200);
    }

    if (!note) {
      throw new Error(`Failed to capture note detail: ${targetUrl}`);
    }

    return {
      url: targetUrl,
      note: normalizeDetailNote(note),
      comments: normalizeComments(comments ?? []),
    };
  } finally {
    page.off('response', handleResponse);
  }
}

export async function getFeedDetails(session: RednoteSession, urls: string[]): Promise<FeedDetailResult> {
  const page = await getOrCreateXiaohongshuPage(session);
  const items: RednoteFeedDetailItem[] = [];
  for (const url of urls) {
    const normalizedUrl = normalizeFeedDetailUrl(url);
    validateFeedDetailUrl(normalizedUrl);
    items.push(await captureFeedDetail(page, normalizedUrl));
  }

  return {
    ok: true,
    detail: {
      fetchedAt: new Date().toISOString(),
      total: items.length,
      items,
    },
  };
}

function writeFeedDetailOutput(result: FeedDetailResult, format: FeedDetailFormat) {
  if (format === 'json') {
    printJson(result);
    return;
  }

  process.stdout.write(renderDetailMarkdown(result.detail.items));
}

export async function runGetFeedDetailCommand(values: FeedDetailCliValues = { urls: [], format: 'md' }) {
  if (values.help) {
    printGetFeedDetailHelp();
    return;
  }
  if (values.urls.length === 0) {
    throw new Error('Missing required option: --url');
  }

  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'fetching feed detail', session);
    const result = await getFeedDetails(session, values.urls);
    writeFeedDetailOutput(result, values.format);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseGetFeedDetailCliArgs(process.argv.slice(2));
  await runGetFeedDetailCommand(values);
}

runCli(import.meta.url, main);
