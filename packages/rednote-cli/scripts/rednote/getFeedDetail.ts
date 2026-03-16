#!/usr/bin/env -S node --experimental-strip-types

import * as cheerio from 'cheerio';
import { parseArgs } from 'node:util';
import vm from 'node:vm';
import type { Page, Response } from 'playwright-core';
import { runCli } from '../utils/browser-cli.ts';
import { simulateMouseMove, simulateMousePresence, simulateMouseWheel } from '../utils/mouse-helper.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn, type RednoteSession } from './checkLogin.ts';
import { ensureJsonSavePath, renderJsonSaveSummary, resolveJsonSavePath, writeJsonFile } from './output-format.ts';
import { findPersistedPostUrlByRecordId, initializeRednoteDatabase, persistFeedDetail } from './persistence.ts';

export type FeedDetailFormat = 'json' | 'md';

export type FeedDetailCliValues = {
  instance?: string;
  urls: string[];
  ids: string[];
  format: FeedDetailFormat;
  comments?: number | null;
  savePath?: string;
  help?: boolean;
};

export type RednoteComment = {
  content: string | null;
  userId: string | null;
  nickname: string | null;
  create_time: string | number | null;
  like_count: string | number | null;
  sub_comment_count: number | null;
};

export type RednoteDetailNote = {
  noteId: string | null;
  title: string | null;
  desc: string | null;
  type: string | null;
  liked: boolean | null;
  likedCount: string | null;
  commentCount: string | null;
  collected: boolean | null;
  collectedCount: string | null;
  shareCount: string | null;
  followed: boolean | null;
  tagList: string[];
  imageList: string[];
  video: string | null;
};

export type RednoteFeedDetailItem = {
  url: string;
  note: RednoteDetailNote;
  comments?: RednoteComment[];
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
  npx -y @skills-store/rednote get-feed-detail [--instance NAME] [--url URL] [--url URL] [--id ID] [--id ID] [--comments [COUNT]] [--format md|json] [--save PATH]
  node --experimental-strip-types ./scripts/rednote/getFeedDetail.ts --instance NAME [--url URL] [--url URL] [--id ID] [--id ID] [--comments [COUNT]] [--format md|json] [--save PATH]
  bun ./scripts/rednote/getFeedDetail.ts --instance NAME [--url URL] [--url URL] [--id ID] [--id ID] [--comments [COUNT]] [--format md|json] [--save PATH]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --url URL         Optional. Xiaohongshu explore url, repeatable
  --id ID           Optional. Database record id from home/search output, repeatable
  --comments [COUNT]  Optional. Include comment data. When COUNT is provided, scroll \`.note-scroller\` until COUNT comments, the end, or timeout
  --format FORMAT   Output format: md | json. Default: md
  --save PATH       Required when --format json is used. Saves the selected result array as JSON
  -h, --help        Show this help
`);
}

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

function parseCommentsValue(value: string | undefined) {
  if (!value) {
    throw new Error('Missing value for --comments');
  }

  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid --comments value: ${String(value)}`);
  }

  return parsed;
}

export function parseGetFeedDetailCliArgs(argv: string[]): FeedDetailCliValues {
  const values: FeedDetailCliValues = {
    urls: [],
    ids: [],
    format: 'md',
    comments: undefined,
    help: false,
  };

  const positionals: string[] = [];

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

    if (withEquals?.key === '--url') {
      values.urls.push(withEquals.value);
      continue;
    }

    if (arg === '--url') {
      const nextArg = argv[index + 1];
      if (!nextArg || nextArg.startsWith('-')) {
        throw new Error('Missing required option value: --url');
      }
      values.urls.push(nextArg);
      index += 1;
      continue;
    }

    if (withEquals?.key === '--id') {
      values.ids.push(withEquals.value);
      continue;
    }

    if (arg === '--id') {
      const nextArg = argv[index + 1];
      if (!nextArg || nextArg.startsWith('-')) {
        throw new Error('Missing required option value: --id');
      }
      values.ids.push(nextArg);
      index += 1;
      continue;
    }

    if (withEquals?.key === '--format') {
      values.format = withEquals.value as FeedDetailFormat;
      continue;
    }

    if (arg === '--format') {
      values.format = argv[index + 1] as FeedDetailFormat;
      index += 1;
      continue;
    }

    if (withEquals?.key === '--comments') {
      values.comments = withEquals.value ? parseCommentsValue(withEquals.value) : null;
      continue;
    }

    if (arg === '--comments') {
      const nextArg = argv[index + 1];
      if (nextArg && !nextArg.startsWith('-')) {
        values.comments = parseCommentsValue(nextArg);
        index += 1;
      } else {
        values.comments = null;
      }
      continue;
    }

    if (withEquals?.key === '--save') {
      values.savePath = withEquals.value;
      continue;
    }

    if (arg === '--save') {
      values.savePath = argv[index + 1];
      index += 1;
      continue;
    }

    positionals.push(arg);
  }

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  if (values.format !== 'md' && values.format !== 'json') {
    throw new Error(`Invalid --format value: ${String(values.format)}`);
  }

  return values;
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

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  return session.page;
}

function extractVideoUrl(note: any) {
  const streams = Object.values(note?.video?.media?.stream ?? {}) as any[];
  const firstAvailable = streams.find((items) => Array.isArray(items) && items.length > 0);
  return firstAvailable?.[0]?.backupUrls?.[0] ?? null;
}

const COMMENTS_CONTAINER_SELECTOR = '.note-scroller';
const COMMENT_SCROLL_TIMEOUT_MS = 20_000;
const COMMENT_SCROLL_IDLE_LIMIT = 4;

function hasCommentsEnabled(comments: FeedDetailCliValues['comments']) {
  return comments !== undefined;
}

function buildCommentKey(comment: any) {
  return String(
    comment?.id
      ?? comment?.commentId
      ?? comment?.comment_id
      ?? `${comment?.userInfo?.userId ?? comment?.user_info?.user_id ?? 'unknown'}:${comment?.createTime ?? comment?.create_time ?? 'unknown'}:${comment?.content ?? ''}`
  );
}

function getCommentCount(commentsMap: Map<string, any>) {
  return commentsMap.size;
}

async function scrollCommentsContainer(page: Page, targetCount: number, getCount: () => number) {
  const container = page.locator(COMMENTS_CONTAINER_SELECTOR).first();
  const visible = await container.isVisible().catch(() => false);
  if (!visible) {
    return;
  }

  await container.scrollIntoViewIfNeeded().catch(() => {});
  await simulateMouseMove(page, { locator: container, settleMs: 100 }).catch(() => {});

  const getMetrics = async () => await container.evaluate((element) => {
    const htmlElement = element as HTMLElement;
    const atBottom = htmlElement.scrollTop + htmlElement.clientHeight >= htmlElement.scrollHeight - 8;
    return {
      scrollTop: htmlElement.scrollTop,
      scrollHeight: htmlElement.scrollHeight,
      clientHeight: htmlElement.clientHeight,
      atBottom,
    };
  }).catch(() => null);

  const deadline = Date.now() + COMMENT_SCROLL_TIMEOUT_MS;
  let idleRounds = 0;

  while (Date.now() < deadline) {
    if (getCount() >= targetCount) {
      return;
    }

    const beforeMetrics = await getMetrics();
    if (!beforeMetrics) {
      return;
    }

    const beforeCount = getCount();
    const delta = Math.max(Math.floor(beforeMetrics.clientHeight * 0.85), 480);
    await simulateMouseWheel(page, { locator: container, deltaY: delta, moveBeforeScroll: false, settleMs: 900 }).catch(() => {});

    const afterMetrics = await getMetrics();
    await page.waitForTimeout(400);
    const afterCount = getCount();

    const countChanged = afterCount > beforeCount;
    const scrollMoved = Boolean(afterMetrics) && afterMetrics.scrollTop > beforeMetrics.scrollTop;
    const reachedBottom = Boolean(afterMetrics?.atBottom);

    if (countChanged || scrollMoved) {
      idleRounds = 0;
      continue;
    }

    idleRounds += 1;
    if ((reachedBottom && idleRounds >= 2) || idleRounds >= COMMENT_SCROLL_IDLE_LIMIT) {
      return;
    }
  }
}

function normalizeDetailNote(note: any): RednoteDetailNote {
  return {
    noteId: note?.noteId ?? null,
    title: note?.title ?? null,
    desc: note?.desc ?? null,
    type: note?.type ?? null,
    liked: note?.interactInfo?.liked ?? null,
    likedCount: note?.interactInfo?.likedCount ?? null,
    commentCount: note?.interactInfo?.commentCount ?? null,
    collected: note?.interactInfo?.collected ?? null,
    collectedCount: note?.interactInfo?.collectedCount ?? null,
    shareCount: note?.interactInfo?.shareCount ?? null,
    followed: note?.interactInfo?.followed ?? null,
    tagList: Array.isArray(note?.tagList)
      ? note.tagList.map((tag: any) => tag?.name ?? null).filter((tag: string | null): tag is string => Boolean(tag))
      : [],
    imageList: Array.isArray(note?.imageList)
      ? note.imageList
          .map((image: any) => image?.urlDefault ?? null)
          .filter((imageUrl: string | null): imageUrl is string => Boolean(imageUrl))
      : [],
    video: extractVideoUrl(note),
  };
}

function normalizeComments(comments: any[]): RednoteComment[] {
  return comments.map((comment) => ({
    content: comment?.content ?? null,
    userId: comment?.userInfo?.userId ?? comment?.user_info?.user_id ?? null,
    nickname: comment?.userInfo?.nickname ?? comment?.user_info?.nickname ?? null,
    create_time: comment?.createTime ?? comment?.create_time ?? null,
    like_count: comment?.likeCount ?? comment?.like_count ?? comment?.interactInfo?.likedCount ?? null,
    sub_comment_count: typeof (comment?.subCommentCount ?? comment?.sub_comment_count) === 'string'
      ? (comment?.subCommentCount ?? comment?.sub_comment_count)
      : null,
  }));
}

function formatDetailField(value: string | number | boolean | null | undefined) {
  return value ?? '';
}

function renderDetailMarkdown(items: RednoteFeedDetailItem[], includeComments = false) {
  if (items.length === 0) {
    return 'No feed details were captured.\n';
  }

  return `${items.map((item) => {
    const lines: string[] = [];

    lines.push('## Note');
    lines.push('');
    lines.push(`- Title: ${formatDetailField(item.note.title)}`);
    lines.push(`- Liked: ${formatDetailField(item.note.liked)}`);
    lines.push(`- Collected: ${formatDetailField(item.note.collected)}`);
    lines.push(`- LikedCount: ${formatDetailField(item.note.likedCount)}`);
    lines.push(`- CommentCount: ${formatDetailField(item.note.commentCount)}`);
    lines.push(`- CollectedCount: ${formatDetailField(item.note.collectedCount)}`);
    lines.push(`- ShareCount: ${formatDetailField(item.note.shareCount)}`);
    lines.push(`- Tags: ${item.note.tagList.map((tag) => `#${tag}`).join(' ')}`);
    lines.push('');
    lines.push('## Content');
    lines.push('');
    lines.push(item.note.desc ?? '');

    if (item.note.imageList.length > 0 || item.note.video) {
      lines.push('');
      lines.push('## Media');
      lines.push('');

      item.note.imageList.forEach((imageUrl, index) => {
        lines.push(`- Image${index + 1}: ${imageUrl}`);
      });

      if (item.note.video) {
        lines.push(`- Video: ${item.note.video}`);
      }
    }

    if (includeComments) {
      lines.push('');
      lines.push('## Comments');
      lines.push('');

      if (!item.comments || item.comments.length === 0) {
        lines.push('- Comments not found');
      } else {
        item.comments.forEach((comment) => {
          const prefix = comment.nickname ? `${comment.nickname}: ` : '';
          lines.push(`- ${prefix}${comment.content ?? ''}`);
        });
      }
    }

    return lines.join('\n');
  }).join('\n\n---\n\n')}\n`;
}

async function captureFeedDetail(
  page: Page,
  targetUrl: string,
  commentsOption: FeedDetailCliValues['comments'] = undefined,
  instanceName?: string,
): Promise<RednoteFeedDetailItem> {
  const includeComments = hasCommentsEnabled(commentsOption);
  const commentsTarget = typeof commentsOption === 'number' ? commentsOption : null;
  let note: any = null;
  let commentsLoaded = !includeComments;
  const commentsMap = new Map<string, any>();

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
      } else if (includeComments && url.href.includes('comment/page?')) {
        const data = await response.json() as { data?: { comments?: any[] } };
        const nextComments = Array.isArray(data?.data?.comments) ? data.data.comments : [];
        commentsLoaded = true;
        for (const comment of nextComments) {
          commentsMap.set(buildCommentKey(comment), comment);
        }
      }
    } catch {
    }
  };

  page.on('response', handleResponse);
  try {
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });
    await simulateMousePresence(page);

    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      if (note && commentsLoaded) {
        break;
      }
      await page.waitForTimeout(200);
    }

    if (!note) {
      throw new Error(`Failed to capture note detail: ${targetUrl}`);
    }

    if (includeComments && commentsTarget) {
      await scrollCommentsContainer(page, commentsTarget, () => getCommentCount(commentsMap));
    }

    const item = {
      url: targetUrl,
      note: normalizeDetailNote(note),
      ...(includeComments ? { comments: normalizeComments([...commentsMap.values()]) } : {}),
    };

    if (instanceName) {
      await persistFeedDetail({
        instanceName,
        url: targetUrl,
        note: item.note,
        rawNote: note,
        rawComments: includeComments ? [...commentsMap.values()] : [],
      });
    }

    await simulateMousePresence(page);
    return item;
  } finally {
    page.off('response', handleResponse);
  }
}

export async function getFeedDetails(
  session: RednoteSession,
  urls: string[],
  commentsOption: FeedDetailCliValues['comments'] = undefined,
  instanceName?: string,
): Promise<FeedDetailResult> {
  const page = await getOrCreateXiaohongshuPage(session);
  const items: RednoteFeedDetailItem[] = [];
  for (const url of urls) {
    items.push(await captureFeedDetail(page, url, commentsOption, instanceName));
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

async function resolveFeedDetailUrls(values: FeedDetailCliValues, instanceName?: string) {
  const urls = [...values.urls];

  if (values.ids.length === 0) {
    return urls;
  }

  if (!instanceName) {
    throw new Error('The --id option requires an instance-backed session.');
  }

  for (const id of values.ids) {
    const url = await findPersistedPostUrlByRecordId(instanceName, id);
    if (!url) {
      throw new Error(`No saved post url found for id: ${id}`);
    }
    urls.push(url);
  }

  return urls;
}

function selectFeedDetailOutput(result: FeedDetailResult) {
  return result.detail.items;
}

function writeFeedDetailOutput(result: FeedDetailResult, values: FeedDetailCliValues) {
  const output = selectFeedDetailOutput(result);

  if (values.format === 'json') {
    const savedPath = resolveJsonSavePath(values.savePath);
    writeJsonFile(output, savedPath);
    process.stdout.write(renderJsonSaveSummary(savedPath, output));
    return;
  }

  process.stdout.write(renderDetailMarkdown(result.detail.items, hasCommentsEnabled(values.comments)));
}

export async function runGetFeedDetailCommand(values: FeedDetailCliValues = { urls: [], ids: [], format: 'md' }) {
  if (values.help) {
    printGetFeedDetailHelp();
    return;
  }

  ensureJsonSavePath(values.format, values.savePath);

  if (values.urls.length === 0 && values.ids.length === 0) {
    throw new Error('Missing required option: --url or --id');
  }

  await initializeRednoteDatabase();

  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'fetching feed detail', session);
    const urls = await resolveFeedDetailUrls(values, target.instanceName);
    const result = await getFeedDetails(session, urls, values.comments, target.instanceName);
    writeFeedDetailOutput(result, values);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseGetFeedDetailCliArgs(process.argv.slice(2));
  await runGetFeedDetailCommand(values);
}

runCli(import.meta.url, main);
