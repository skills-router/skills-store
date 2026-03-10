#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import type { Locator, Page } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import {
  createRednoteSession,
  disconnectRednoteSession,
  ensureRednoteLoggedIn,
  type RednoteSession,
} from './checkLogin.ts';
import { commentOnFeed } from './comment.ts';
import { getFeedDetails } from './getFeedDetail.ts';

export type InteractAction = 'like' | 'collect' | 'comment';

export type InteractCliValues = {
  instance?: string;
  url?: string;
  action?: string;
  content?: string;
  help?: boolean;
};

export type InteractResult = {
  ok: true;
  message: string;
};

const INTERACT_CONTAINER_SELECTOR = '.interact-container';
const LIKE_WRAPPER_SELECTOR = `${INTERACT_CONTAINER_SELECTOR} .like-wrapper`;
const COLLECT_WRAPPER_SELECTOR = `${INTERACT_CONTAINER_SELECTOR} .collect-wrapper, ${INTERACT_CONTAINER_SELECTOR} #note-page-collect-board-guide`;

function printInteractHelp() {
  process.stdout.write(`rednote interact

Usage:
  npx -y @skills-store/rednote interact [--instance NAME] --url URL --action like|collect|comment [--content TEXT]
  node --experimental-strip-types ./scripts/rednote/interact.ts --instance NAME --url URL --action like|collect|comment [--content TEXT]
  bun ./scripts/rednote/interact.ts --instance NAME --url URL --action like|collect|comment [--content TEXT]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --url URL         Required. Xiaohongshu explore url
  --action ACTION   Required. like | collect | comment
  --content TEXT    Required only when --action comment
  -h, --help        Show this help
`);
}

export function parseInteractCliArgs(argv: string[]): InteractCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      url: { type: 'string' },
      action: { type: 'string' },
      content: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  return {
    instance: values.instance,
    url: values.url,
    action: values.action,
    content: values.content,
    help: values.help,
  };
}

function ensureNonEmpty(value: string | undefined, optionName: string) {
  const normalized = value?.trim();
  if (!normalized) {
    throw new Error(`Missing required option: ${optionName}`);
  }

  return normalized;
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

function resolveInteractAction(action: string | undefined): InteractAction {
  const normalized = action?.trim().toLowerCase();
  if (!normalized) {
    throw new Error('Missing required option: --action');
  }

  if (normalized === 'like' || normalized === 'collect' || normalized === 'comment') {
    return normalized;
  }

  if (normalized === 'favorite') {
    return 'collect';
  }

  throw new Error(`Invalid --action value: ${String(action)}. Expected like | collect | comment`);
}

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  return session.page;
}

async function findVisibleLocator(locator: Locator, timeoutMs = 5_000) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const count = await locator.count();
    for (let index = 0; index < count; index += 1) {
      const candidate = locator.nth(index);
      if (await candidate.isVisible().catch(() => false)) {
        return candidate;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  return null;
}

async function requireVisibleLocator(locator: Locator, errorMessage: string, timeoutMs = 5_000) {
  const visibleLocator = await findVisibleLocator(locator, timeoutMs);
  if (!visibleLocator) {
    throw new Error(errorMessage);
  }

  return visibleLocator;
}

async function waitForInteractContainer(page: Page) {
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(500);
  await requireVisibleLocator(
    page.locator(INTERACT_CONTAINER_SELECTOR),
    '未找到互动工具栏，请确认帖子详情页已正确加载。',
    15_000,
  );
}

function getActionErrorMessage(action: 'like' | 'collect') {
  return action === 'like'
    ? '未找到点赞按钮，请确认帖子详情页已正确加载。'
    : '未找到收藏按钮，请确认帖子详情页已正确加载。';
}

async function ensureActionApplied(page: Page, action: 'like' | 'collect', alreadyActive: boolean) {
  if (alreadyActive) {
    return true;
  }

  const locator = page.locator(action === 'like' ? LIKE_WRAPPER_SELECTOR : COLLECT_WRAPPER_SELECTOR);
  const visibleLocator = await requireVisibleLocator(locator, getActionErrorMessage(action), 15_000);

  await visibleLocator.scrollIntoViewIfNeeded();
  await visibleLocator.click({ force: true });

  await page.waitForFunction(
    ({ selector, currentAction }) => {
      const nodes = [...document.querySelectorAll(selector)];
      const target = nodes.find((candidate) => candidate instanceof HTMLElement && candidate.offsetParent !== null);
      if (!(target instanceof HTMLElement)) {
        return false;
      }

      const classNames = [...target.classList].map((item) => item.toLowerCase());
      if (classNames.includes(`${currentAction}-active`) || classNames.includes('active') || classNames.some((item) => item.endsWith('-active'))) {
        return true;
      }

      const iconRefs = [...target.querySelectorAll('use')]
        .map((node) => (node.getAttribute('xlink:href') ?? node.getAttribute('href') ?? '').toLowerCase())
        .filter(Boolean)
        .join(' ');

      if (currentAction === 'like') {
        return iconRefs.includes('liked') || iconRefs.includes('like-filled') || iconRefs.includes('like_fill');
      }

      return iconRefs.includes('collected') || iconRefs.includes('collect-filled') || iconRefs.includes('collect_fill');
    },
    {
      selector: action === 'like' ? LIKE_WRAPPER_SELECTOR : COLLECT_WRAPPER_SELECTOR,
      currentAction: action,
    },
    { timeout: 10_000 },
  );

  return false;
}

export async function interactWithFeed(
  session: RednoteSession,
  url: string,
  action: InteractAction,
  content?: string,
): Promise<InteractResult> {
  if (action === 'comment') {
    const normalizedContent = ensureNonEmpty(content, '--content');
    const commentResult = await commentOnFeed(session, url, normalizedContent);
    return {
      ok: true,
      message: `Comment posted: ${url}`,
    };
  }

  validateFeedDetailUrl(url);
  const detailResult = await getFeedDetails(session, [url]);
  const detailItem = detailResult.detail.items[0];
  if (!detailItem) {
    throw new Error(`Failed to load feed detail: ${url}`);
  }

  const alreadyActive = action === 'like'
    ? detailItem.note.interactInfo.liked === true
    : detailItem.note.interactInfo.collected === true;

  const page = await getOrCreateXiaohongshuPage(session);
  await waitForInteractContainer(page);
  await ensureActionApplied(page, action, alreadyActive);

  const message = alreadyActive
    ? `${action === 'like' ? 'Like' : 'Collect'} already active: ${url}`
    : `${action === 'like' ? 'Like' : 'Collect'} completed: ${url}`;

  return {
    ok: true,
    message,
  };
}

export async function runInteractCommand(values: InteractCliValues = {}) {
  if (values.help) {
    printInteractHelp();
    return;
  }

  const url = ensureNonEmpty(values.url, '--url');
  const action = resolveInteractAction(values.action);
  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, `performing ${action} interact`, session);
    const result = await interactWithFeed(session, url, action, values.content);
    printJson(result);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseInteractCliArgs(process.argv.slice(2));
  await runInteractCommand(values);
}

runCli(import.meta.url, main);
