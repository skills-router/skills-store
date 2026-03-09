#!/usr/bin/env -S node --experimental-strip-types

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import type { Page } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget, type RednoteStatusTarget } from './status.ts';
import {
  createRednoteSession,
  disconnectRednoteSession,
  ensureRednoteLoggedIn,
  type RednoteSession,
} from './checkLogin.ts';

const REDNOTE_EXPLORE_URL = 'https://www.xiaohongshu.com/explore';
const CREATOR_HOME_URL = 'https://creator.xiaohongshu.com/new/home';
const CREATOR_SERVICE_SELECTOR = 'a.link[href="//creator.xiaohongshu.com/?source=official"]';
const MAX_IMAGE_COUNT = 15;

export type PublishType = 'video' | 'image' | 'article';

export type PublishCliValues = {
  instance?: string;
  type?: PublishType;
  title?: string;
  description?: string;
  tags: string[];
  videoPath?: string;
  imagePaths: string[];
  markdownPath?: string;
  publishNow: boolean;
  help?: boolean;
};

export type ResolvedPublishPayload =
  | {
      type: 'video';
      title: string;
      description: string;
      tags: string[];
      draft: boolean;
      videoPath: string;
    }
  | {
      type: 'image';
      title: string;
      description: string;
      tags: string[];
      draft: boolean;
      imagePaths: string[];
      coverImagePath: string;
    }
  | {
      type: 'article';
      title: string;
      draft: boolean;
      markdownPath: string;
      markdownContent: string;
    };

export type PublishResult = {
  ok: true;
  instance: {
    scope: 'default' | 'custom';
    name: string;
    browser: RednoteStatusTarget['browser'];
    userDataDir: string | null;
    source: RednoteStatusTarget['source'];
    lastConnect: boolean;
  };
  publish: {
    pageUrl: string;
    creatorHomeUrl: string;
    clickedCreatorService: boolean;
    reusedCreatorHome: boolean;
    openedInNewPage: boolean;
    payload: {
      type: PublishType;
      title: string;
      description: string | null;
      tags: string[];
      draft: boolean;
      assetCount: number;
      coverImagePath: string | null;
      videoPath: string | null;
      imagePaths: string[];
      markdownPath: string | null;
      markdownLength: number;
    };
    message: string;
  };
};

function printPublishHelp() {
  process.stdout.write(`rednote publish

Usage:
  npx -y @skills-store/rednote publish --type video --video ./video.mp4 --title 标题 --description 描述 [--tag 穿搭] [--tag 日常] [--publish] [--instance NAME]
  npx -y @skills-store/rednote publish --type image --image ./1.jpg --image ./2.jpg --title 标题 --description 描述 [--tag 探店] [--publish] [--instance NAME]
  npx -y @skills-store/rednote publish --type article --title 标题 --markdown ./article.md [--publish] [--instance NAME]
  node --experimental-strip-types ./scripts/rednote/publish.ts ...
  bun ./scripts/rednote/publish.ts ...

Options:
  --instance NAME      Optional. Defaults to the saved lastConnect instance
  --type TYPE          Required. video | image | article
  --title TEXT         Required. 发布标题
  --description TEXT   视频/图文必填。发布描述
  --tag TEXT           可选。重复传入多个标签，例如 --tag 穿搭 --tag OOTD
  --video PATH         视频模式必填。只能传 1 个视频文件
  --image PATH         图文模式必填。重复传入多张图片，最多 ${MAX_IMAGE_COUNT} 张，首张为首图
  --markdown PATH      长文模式必填。Markdown 文件路径
  --publish            立即发布。不传时默认保存草稿
  -h, --help           Show this help
`);
}

function isCreatorHomeUrl(url: string) {
  return url === CREATOR_HOME_URL || url.startsWith(`${CREATOR_HOME_URL}?`) || url.startsWith(`${CREATOR_HOME_URL}#`);
}

function ensureNonEmpty(value: string | undefined, optionName: string) {
  const normalized = value?.trim();
  if (!normalized) {
    throw new Error(`Missing required option: ${optionName}`);
  }
  return normalized;
}

function normalizeTags(tags: string[]) {
  const normalizedTags = tags
    .map((tag) => tag.trim())
    .filter(Boolean)
    .map((tag) => tag.replace(/^#+/, ''))
    .filter(Boolean);

  return [...new Set(normalizedTags)];
}

function resolveExistingFile(filePath: string, optionName: string) {
  const resolvedPath = path.resolve(filePath);
  let stat: fs.Stats;

  try {
    stat = fs.statSync(resolvedPath);
  } catch {
    throw new Error(`${optionName} file not found: ${resolvedPath}`);
  }

  if (!stat.isFile()) {
    throw new Error(`${optionName} must point to a file: ${resolvedPath}`);
  }

  return resolvedPath;
}

export function parsePublishCliArgs(argv: string[]): PublishCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      type: { type: 'string' },
      title: { type: 'string' },
      description: { type: 'string' },
      tag: { type: 'string', multiple: true },
      video: { type: 'string' },
      image: { type: 'string', multiple: true },
      markdown: { type: 'string' },
      publish: { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  const publishType = values.type;
  if (publishType && publishType !== 'video' && publishType !== 'image' && publishType !== 'article') {
    throw new Error(`Invalid --type value: ${String(publishType)}`);
  }

  return {
    instance: values.instance,
    type: publishType,
    title: values.title,
    description: values.description,
    tags: values.tag ?? [],
    videoPath: values.video,
    imagePaths: values.image ?? [],
    markdownPath: values.markdown,
    publishNow: values.publish ?? false,
    help: values.help,
  };
}

export function resolvePublishPayload(values: PublishCliValues): ResolvedPublishPayload {
  const type = values.type;
  if (!type) {
    throw new Error('Missing required option: --type');
  }

  const title = ensureNonEmpty(values.title, '--title');
  const tags = normalizeTags(values.tags);
  const draft = !values.publishNow;

  if (type === 'video') {
    const description = ensureNonEmpty(values.description, '--description');
    const videoPath = ensureNonEmpty(values.videoPath, '--video');

    if (values.imagePaths.length > 0) {
      throw new Error('Do not combine --type video with --image');
    }
    if (values.markdownPath) {
      throw new Error('Do not combine --type video with --markdown');
    }

    return {
      type,
      title,
      description,
      tags,
      draft,
      videoPath: resolveExistingFile(videoPath, '--video'),
    };
  }

  if (type === 'image') {
    const description = ensureNonEmpty(values.description, '--description');

    if (values.videoPath) {
      throw new Error('Do not combine --type image with --video');
    }
    if (values.markdownPath) {
      throw new Error('Do not combine --type image with --markdown');
    }
    if (values.imagePaths.length === 0) {
      throw new Error('Missing required option: --image');
    }
    if (values.imagePaths.length > MAX_IMAGE_COUNT) {
      throw new Error(`Too many images: received ${values.imagePaths.length}, maximum is ${MAX_IMAGE_COUNT}`);
    }

    const imagePaths = values.imagePaths.map((imagePath) => resolveExistingFile(imagePath, '--image'));

    return {
      type,
      title,
      description,
      tags,
      draft,
      imagePaths,
      coverImagePath: imagePaths[0],
    };
  }

  if (values.videoPath) {
    throw new Error('Do not combine --type article with --video');
  }
  if (values.imagePaths.length > 0) {
    throw new Error('Do not combine --type article with --image');
  }
  if (values.description?.trim()) {
    throw new Error('Do not combine --type article with --description');
  }
  if (tags.length > 0) {
    throw new Error('Do not combine --type article with --tag');
  }

  const markdownPath = ensureNonEmpty(values.markdownPath, '--markdown');
  const resolvedMarkdownPath = resolveExistingFile(markdownPath, '--markdown');
  const markdownContent = fs.readFileSync(resolvedMarkdownPath, 'utf8').trim();

  if (!markdownContent) {
    throw new Error(`--markdown file is empty: ${resolvedMarkdownPath}`);
  }

  return {
    type,
    title,
    draft,
    markdownPath: resolvedMarkdownPath,
    markdownContent,
  };
}

function summarizePayload(payload: ResolvedPublishPayload): PublishResult['publish']['payload'] {
  if (payload.type === 'video') {
    return {
      type: payload.type,
      title: payload.title,
      description: payload.description,
      tags: payload.tags,
      draft: payload.draft,
      assetCount: 1,
      coverImagePath: null,
      videoPath: payload.videoPath,
      imagePaths: [],
      markdownPath: null,
      markdownLength: 0,
    };
  }

  if (payload.type === 'image') {
    return {
      type: payload.type,
      title: payload.title,
      description: payload.description,
      tags: payload.tags,
      draft: payload.draft,
      assetCount: payload.imagePaths.length,
      coverImagePath: payload.coverImagePath,
      videoPath: null,
      imagePaths: payload.imagePaths,
      markdownPath: null,
      markdownLength: 0,
    };
  }

  return {
    type: payload.type,
    title: payload.title,
    description: null,
    tags: [],
    draft: payload.draft,
    assetCount: 0,
    coverImagePath: null,
    videoPath: null,
    imagePaths: [],
    markdownPath: payload.markdownPath,
    markdownLength: payload.markdownContent.length,
  };
}

function toPublishResult(
  target: RednoteStatusTarget,
  publish: PublishResult['publish'],
): PublishResult {
  return {
    ok: true,
    instance: {
      scope: target.scope,
      name: target.instanceName,
      browser: target.browser,
      userDataDir: target.userDataDir,
      source: target.source,
      lastConnect: target.lastConnect,
    },
    publish,
  };
}

function getSessionPages(session: RednoteSession) {
  const pages = [session.page, ...session.browserContext.pages()];
  return [...new Set(pages)];
}

async function findCreatorServicePage(session: RednoteSession) {
  for (const page of getSessionPages(session)) {
    if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
      continue;
    }

    const creatorServiceLink = page.locator(CREATOR_SERVICE_SELECTOR).filter({ hasText: '创作服务' });
    if ((await creatorServiceLink.count()) > 0) {
      return page;
    }
  }

  return null;
}

async function resolvePublishPage(session: RednoteSession) {
  const existingCreatorHomePage = getSessionPages(session).find((page) => isCreatorHomeUrl(page.url()));
  if (existingCreatorHomePage) {
    return {
      page: existingCreatorHomePage,
      reusedCreatorHome: true,
    };
  }

  const existingCreatorServicePage = await findCreatorServicePage(session);
  if (existingCreatorServicePage) {
    return {
      page: existingCreatorServicePage,
      reusedCreatorHome: false,
    };
  }

  const page = session.page;
  if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
    await page.goto(REDNOTE_EXPLORE_URL, {
      waitUntil: 'domcontentloaded',
    });
  }

  await page.waitForTimeout(1_500);

  return {
    page,
    reusedCreatorHome: isCreatorHomeUrl(page.url()),
  };
}

async function waitForCreatorHome(page: Page) {
  await page.waitForURL((url) => isCreatorHomeUrl(url.toString()), {
    timeout: 15_000,
  });
  await page.waitForLoadState('domcontentloaded');
}

export async function openRednotePublish(
  target: RednoteStatusTarget,
  session: RednoteSession,
  payload: ResolvedPublishPayload,
): Promise<PublishResult> {
  const resolved = await resolvePublishPage(session);
  const payloadSummary = summarizePayload(payload);

  if (resolved.reusedCreatorHome || isCreatorHomeUrl(resolved.page.url())) {
    return toPublishResult(target, {
      pageUrl: resolved.page.url(),
      creatorHomeUrl: CREATOR_HOME_URL,
      clickedCreatorService: false,
      reusedCreatorHome: true,
      openedInNewPage: false,
      payload: payloadSummary,
      message: '当前页面已经是创作服务首页，发布参数已校验。',
    });
  }

  const creatorServiceLink = resolved.page.locator(CREATOR_SERVICE_SELECTOR).filter({ hasText: '创作服务' });
  if ((await creatorServiceLink.count()) === 0) {
    throw new Error('未找到“创作服务”入口，请先打开小红书首页并确认账号已登录。');
  }

  const popupPromise = session.browserContext
    .waitForEvent('page', {
      timeout: 3_000,
    })
    .catch(() => null);

  await creatorServiceLink.first().click();

  let targetPage = (await popupPromise) ?? resolved.page;
  let openedInNewPage = targetPage !== resolved.page;

  try {
    await waitForCreatorHome(targetPage);
  } catch {
    const existingCreatorHomePage = getSessionPages(session).find((page) => isCreatorHomeUrl(page.url()));
    if (!existingCreatorHomePage) {
      throw new Error(`点击“创作服务”后，未跳转到 ${CREATOR_HOME_URL}`);
    }

    targetPage = existingCreatorHomePage;
    openedInNewPage = targetPage !== resolved.page;
  }

  return toPublishResult(target, {
    pageUrl: targetPage.url(),
    creatorHomeUrl: CREATOR_HOME_URL,
    clickedCreatorService: true,
    reusedCreatorHome: false,
    openedInNewPage,
    payload: payloadSummary,
    message: '已进入创作服务首页，发布参数已校验。',
  });
}

export async function runPublishCommand(values: PublishCliValues) {
  if (values.help) {
    printPublishHelp();
    return;
  }

  const payload = resolvePublishPayload(values);
  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'publishing content', session);
    const result = await openRednotePublish(target, session, payload);
    printJson(result);
  } finally {
    disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parsePublishCliArgs(process.argv.slice(2));

  if (values.help) {
    printPublishHelp();
    return;
  }

  await runPublishCommand(values);
}

runCli(import.meta.url, main);
