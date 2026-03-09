#!/usr/bin/env -S node --experimental-strip-types

import type { Page, Response } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn, type RednoteSession } from './checkLogin.ts';
import type { RednotePost } from './post-types.ts';
import {
  parseOutputCliArgs,
  renderPostsMarkdown,
  resolveSavePath,
  writePostsJsonl,
  type OutputCliValues,
} from './output-format.ts';

export interface XHSSearchItem {
  id: string;
  model_type: string;
  xsec_token?: string;
  note_card?: {
    type?: string;
    display_title?: string;
    user?: {
      avatar?: string;
      user_id?: string;
      nickname?: string;
      nick_name?: string;
      xsec_token?: string;
    };
    interact_info?: {
      liked?: boolean;
      liked_count?: string;
      comment_count?: string;
      collected_count?: string;
      shared_count?: string;
    };
    cover?: {
      url_default?: string;
      url_pre?: string;
      url?: string;
      file_id?: string;
      height?: number;
      width?: number;
      info_list?: Array<{
        image_scene?: string;
        url?: string;
      }>;
    };
    corner_tag_info?: Array<{
      type?: string;
      text?: string;
    }>;
    image_list?: Array<{
      width?: number;
      height?: number;
      info_list?: Array<{
        image_scene?: string;
        url?: string;
      }>;
    }>;
    video?: {
      capa?: {
        duration?: number;
      };
    };
  };
}

export type SearchCliValues = OutputCliValues;

export type SearchResult = {
  ok: true;
  search: {
    keyword: string;
    pageUrl: string;
    fetchedAt: string;
    total: number;
    posts: RednotePost[];
    savedPath?: string;
  };
};

export function parseSearchCliArgs(argv: string[]) {
  return parseOutputCliArgs(argv, { includeKeyword: true });
}

function printSearchHelp() {
  process.stdout.write(`rednote search

Usage:
  npx -y @skills-store/rednote search [--instance NAME] --keyword TEXT [--format md|json] [--save [PATH]]
  node --experimental-strip-types ./scripts/rednote/search.ts --instance NAME --keyword TEXT [--format md|json] [--save [PATH]]
  bun ./scripts/rednote/search.ts --instance NAME --keyword TEXT [--format md|json] [--save [PATH]]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --keyword TEXT    Required. Search keyword
  --format FORMAT   Output format: md | json. Default: md
  --save [PATH]     Save posts as JSONL. Uses a default path when PATH is omitted
  -h, --help        Show this help
`);
}

function normalizeSearchPost(item: XHSSearchItem): RednotePost {
  const noteCard = item.note_card ?? {};
  const user = noteCard.user ?? {};
  const interactInfo = noteCard.interact_info ?? {};
  const cover = noteCard.cover ?? {};
  const imageList = Array.isArray(noteCard.image_list) ? noteCard.image_list : [];
  const cornerTagInfo = Array.isArray(noteCard.corner_tag_info) ? noteCard.corner_tag_info : [];
  const xsecToken = item.xsec_token ?? null;

  return {
    id: item.id,
    modelType: item.model_type,
    xsecToken,
    url: xsecToken
      ? `https://www.xiaohongshu.com/explore/${item.id}?xsec_token=${xsecToken}`
      : `https://www.xiaohongshu.com/explore/${item.id}`,
    noteCard: {
      type: noteCard.type ?? null,
      displayTitle: noteCard.display_title ?? null,
      cover: {
        urlDefault: cover.url_default ?? null,
        urlPre: cover.url_pre ?? null,
        url: cover.url ?? null,
        fileId: cover.file_id ?? null,
        width: cover.width ?? null,
        height: cover.height ?? null,
        infoList: Array.isArray(cover.info_list)
          ? cover.info_list.map((info) => ({
              imageScene: info?.image_scene ?? null,
              url: info?.url ?? null,
            }))
          : [],
      },
      user: {
        userId: user.user_id ?? null,
        nickname: user.nickname ?? null,
        nickName: user.nick_name ?? user.nickname ?? null,
        avatar: user.avatar ?? null,
        xsecToken: user.xsec_token ?? null,
      },
      interactInfo: {
        liked: interactInfo.liked ?? false,
        likedCount: interactInfo.liked_count ?? null,
        commentCount: interactInfo.comment_count ?? null,
        collectedCount: interactInfo.collected_count ?? null,
        sharedCount: interactInfo.shared_count ?? null,
      },
      cornerTagInfo: cornerTagInfo.map((tag) => ({
        type: tag?.type ?? null,
        text: tag?.text ?? null,
      })),
      imageList: imageList.map((image) => ({
        width: image?.width ?? null,
        height: image?.height ?? null,
        infoList: Array.isArray(image?.info_list)
          ? image.info_list.map((info) => ({
              imageScene: info?.image_scene ?? null,
              url: info?.url ?? null,
            }))
          : [],
      })),
      video: {
        duration: noteCard.video?.capa?.duration ?? null,
      },
    },
  };
}

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  return session.page;
}

function isJsonContentType(contentType: string | undefined) {
  return typeof contentType === 'string' && contentType.includes('/json');
}

async function collectSearchItems(page: Page, keyword: string) {
  const items = new Map<string, XHSSearchItem>();

  const searchPromise = new Promise<XHSSearchItem[]>((resolve, reject) => {
    const handleResponse = async (response: Response) => {
      try {
        if (response.status() !== 200) {
          return;
        }

        if (response.request().method().toLowerCase() !== 'post') {
          return;
        }

        if (!isJsonContentType(response.headers()['content-type'])) {
          return;
        }

        const data = await response.json() as { success?: boolean; data?: { items?: XHSSearchItem[] } };
        const list = Array.isArray(data?.data?.items) ? data.data.items : null;
        if (!data?.success || !list) {
          return;
        }

        for (const item of list) {
          if (item && item.model_type === 'note' && typeof item.id === 'string') {
            items.set(item.id, item);
          }
        }

        if (items.size > 0) {
          clearTimeout(timeoutId);
          page.off('response', handleResponse);
          resolve([...items.values()]);
        }
      } catch {
      }
    };

    const timeoutId = setTimeout(() => {
      page.off('response', handleResponse);
      reject(new Error(`Timed out waiting for Xiaohongshu search response: ${keyword}`));
    }, 15_000);

    page.on('response', handleResponse);
  });

  if (!page.url().startsWith('https://www.xiaohongshu.com/explore')) {
    await page.goto('https://www.xiaohongshu.com/explore', {
      waitUntil: 'domcontentloaded',
    });
  }

  const searchInput = page.locator('#search-input');
  await searchInput.focus();
  await searchInput.fill(keyword);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);

  return await searchPromise;
}

export async function searchRednotePosts(session: RednoteSession, keyword: string): Promise<SearchResult> {
  const page = await getOrCreateXiaohongshuPage(session);
  const items = await collectSearchItems(page, keyword);
  const posts = items.map(normalizeSearchPost);

  return {
    ok: true,
    search: {
      keyword,
      pageUrl: page.url(),
      fetchedAt: new Date().toISOString(),
      total: posts.length,
      posts,
    },
  };
}

function writeSearchOutput(result: SearchResult, values: SearchCliValues) {
  const posts = result.search.posts;
  let savedPath: string | undefined;

  if (values.saveRequested) {
    savedPath = resolveSavePath('search', values.savePath, result.search.keyword);
    writePostsJsonl(posts, savedPath);
    result.search.savedPath = savedPath;
  }

  if (values.format === 'json') {
    printJson(result);
    return;
  }

  let markdown = renderPostsMarkdown(posts);
  if (savedPath) {
    markdown = `Saved JSONL: ${savedPath}\n\n${markdown}`;
  }
  process.stdout.write(markdown);
}

export async function runSearchCommand(values: SearchCliValues = { format: 'md', saveRequested: false }) {
  if (values.help) {
    printSearchHelp();
    return;
  }


  const keyword = values.keyword?.trim();
  if (!keyword) {
    throw new Error('Missing required option: --keyword');
  }

  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'search', session);
    const result = await searchRednotePosts(session, keyword);
    writeSearchOutput(result, values);
  } finally {
    disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseSearchCliArgs(process.argv.slice(2));

  if (values.help) {
    printSearchHelp();
    return;
  }

  await runSearchCommand(values);
}

runCli(import.meta.url, main);
