#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import type { Page, Response } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import * as cheerio from 'cheerio';
import vm from 'node:vm';
import type { RednotePost } from './post-types.ts';
import {
  parseOutputCliArgs,
  renderPostsMarkdown,
  resolveSavePath,
  writePostsJsonl,
  type OutputCliValues,
} from './output-format.ts';
import { createRednoteSession, disconnectRednoteSession, type RednoteSession } from './checkLogin.ts';

export interface XHSHomeFeedItem {
  id: string;
  modelType: string;
  xsecToken?: string;
  noteCard?: {
    type?: string;
    displayTitle?: string;
    user?: {
      avatar?: string;
      userId?: string;
      nickname?: string;
      nickName?: string;
      xsecToken?: string;
    };
    interactInfo?: {
      liked?: boolean;
      likedCount?: string;
      commentCount?: string;
      collectedCount?: string;
      sharedCount?: string;
    };
    cover?: {
      urlDefault?: string;
      urlPre?: string;
      url?: string;
      fileId?: string;
      height?: number;
      width?: number;
      infoList?: Array<{
        imageScene?: string;
        url?: string;
      }>;
    };
    cornerTagInfo?: Array<{
      type?: string;
      text?: string;
    }>;
    imageList?: Array<{
      width?: number;
      height?: number;
      infoList?: Array<{
        imageScene?: string;
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

export type HomeCliValues = OutputCliValues;

export type HomeResult = {
  ok: true;
  home: {
    pageUrl: string;
    fetchedAt: string;
    total: number;
    posts: RednotePost[];
    savedPath?: string;
  };
};

export function parseHomeCliArgs(argv: string[]) {
  return parseOutputCliArgs(argv);
}

function printHomeHelp() {
  process.stdout.write(`rednote home

Usage:
  npx -y @skills-store/rednote home [--instance NAME] [--format md|json] [--save [PATH]]
  node --experimental-strip-types ./scripts/rednote/home.ts --instance NAME [--format md|json] [--save [PATH]]
  bun ./scripts/rednote/home.ts --instance NAME [--format md|json] [--save [PATH]]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --format FORMAT   Output format: md | json. Default: md
  --save [PATH]     Save posts as JSONL. Uses a default path when PATH is omitted
  -h, --help        Show this help
`);
}

function normalizeHomePost(item: XHSHomeFeedItem): RednotePost {
  const noteCard = item.noteCard ?? {};
  const user = noteCard.user ?? {};
  const interactInfo = noteCard.interactInfo ?? {};
  const cover = noteCard.cover ?? {};
  const imageList = Array.isArray(noteCard.imageList) ? noteCard.imageList : [];
  const cornerTagInfo = Array.isArray(noteCard.cornerTagInfo) ? noteCard.cornerTagInfo : [];
  const xsecToken = item.xsecToken ?? null;

  return {
    id: item.id,
    modelType: item.modelType,
    xsecToken,
    url: xsecToken
      ? `https://www.xiaohongshu.com/explore/${item.id}?xsec_token=${xsecToken}`
      : `https://www.xiaohongshu.com/explore/${item.id}`,
    noteCard: {
      type: noteCard.type ?? null,
      displayTitle: noteCard.displayTitle ?? null,
      cover: {
        urlDefault: cover.urlDefault ?? null,
        urlPre: cover.urlPre ?? null,
        url: cover.url ?? null,
        fileId: cover.fileId ?? null,
        width: cover.width ?? null,
        height: cover.height ?? null,
        infoList: Array.isArray(cover.infoList)
          ? cover.infoList.map((info) => ({
              imageScene: info?.imageScene ?? null,
              url: info?.url ?? null,
            }))
          : [],
      },
      user: {
        userId: user.userId ?? null,
        nickname: user.nickname ?? null,
        nickName: user.nickName ?? user.nickname ?? null,
        avatar: user.avatar ?? null,
        xsecToken: user.xsecToken ?? null,
      },
      interactInfo: {
        liked: interactInfo.liked ?? false,
        likedCount: interactInfo.likedCount ?? null,
        commentCount: interactInfo.commentCount ?? null,
        collectedCount: interactInfo.collectedCount ?? null,
        sharedCount: interactInfo.sharedCount ?? null,
      },
      cornerTagInfo: cornerTagInfo.map((tag) => ({
        type: tag?.type ?? null,
        text: tag?.text ?? null,
      })),
      imageList: imageList.map((image) => ({
        width: image?.width ?? null,
        height: image?.height ?? null,
        infoList: Array.isArray(image?.infoList)
          ? image.infoList.map((info) => ({
              imageScene: info?.imageScene ?? null,
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

async function collectHomeFeedItems(page: Page) {
  const items = new Map<string, XHSHomeFeedItem>();

  const feedPromise = new Promise<XHSHomeFeedItem[]>((resolve, reject) => {
    const handleResponse = async (response: Response) => {
      try {
        if (response.status() !== 200) {
          return;
        }

        if (response.request().method().toLowerCase() !== 'get') {
          return;
        }

        const url = new URL(response.url());
        if (!url.href.endsWith('/explore')) {
          return;
        }

        const html = await response.text();
        const $ = cheerio.load(html);

        $('script').each((_, element) => {
          const scriptContent = $(element).html();
          if (!scriptContent?.includes('window.__INITIAL_STATE__')) {
            return;
          }

          const scriptText = scriptContent.substring(scriptContent.indexOf('=') + 1).trim();
          const sandbox: { info?: { feed?: { feeds?: XHSHomeFeedItem[] } } } = {};
          vm.createContext(sandbox);
          vm.runInContext(`var info = ${scriptText}`, sandbox);

          const feeds = sandbox.info?.feed?.feeds;
          if (!Array.isArray(feeds)) {
            return;
          }

          for (const feed of feeds) {
            if (feed && feed.modelType === 'note' && typeof feed.id === 'string') {
              items.set(feed.id, feed);
            }
          }
        });

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
      reject(new Error('Timed out waiting for Xiaohongshu home feed response'));
    }, 15_000);

    page.on('response', handleResponse);
  });

  if (page.url().startsWith('https://www.xiaohongshu.com/explore')) {
    await page.reload({ waitUntil: 'domcontentloaded' });
  } else {
    await page.goto('https://www.xiaohongshu.com/explore', { waitUntil: 'domcontentloaded' });
  }

  await page.waitForTimeout(500);
  return await feedPromise;
}

export async function getRednoteHomePosts(session: RednoteSession): Promise<HomeResult> {
  const page = await getOrCreateXiaohongshuPage(session);
  const items = await collectHomeFeedItems(page);
  const posts = items.map(normalizeHomePost);

  return {
    ok: true,
    home: {
      pageUrl: page.url(),
      fetchedAt: new Date().toISOString(),
      total: posts.length,
      posts,
    },
  };
}

function writeHomeOutput(result: HomeResult, values: HomeCliValues) {
  const posts = result.home.posts;
  let savedPath: string | undefined;

  if (values.saveRequested) {
    savedPath = resolveSavePath('home', values.savePath);
    writePostsJsonl(posts, savedPath);
    result.home.savedPath = savedPath;
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

export async function runHomeCommand(values: HomeCliValues = { format: 'md', saveRequested: false }) {
  if (values.help) {
    printHomeHelp();
    return;
  }


  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    const result = await getRednoteHomePosts(session);
    writeHomeOutput(result, values);
  } finally {
    disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseHomeCliArgs(process.argv.slice(2));

  if (values.help) {
    printHomeHelp();
    return;
  }

  await runHomeCommand(values);
}

runCli(import.meta.url, main);
