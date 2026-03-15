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
import { getFeedDetails } from './getFeedDetail.ts';

export type InteractAction = 'like' | 'collect' | 'comment';

export type InteractCliValues = {
  instance?: string;
  url?: string;
  like?: boolean;
  collect?: boolean;
  comment?: string;
  help?: boolean;
};

export type InteractResult = {
  ok: true;
  message: string;
};

const INTERACT_CONTAINER_SELECTOR = '.interact-container';
const LIKE_WRAPPER_SELECTOR = `${INTERACT_CONTAINER_SELECTOR} .like-wrapper`;
const COLLECT_WRAPPER_SELECTOR = `${INTERACT_CONTAINER_SELECTOR} .collect-wrapper, ${INTERACT_CONTAINER_SELECTOR} #note-page-collect-board-guide`;
const COMMENT_INPUT_SELECTOR = '#content-textarea[contenteditable="true"]';
const COMMENT_SEND_BUTTON_SELECTOR = 'button.btn.submit';
const COMMENT_SEND_BUTTON_TEXT = '发送';

function printInteractHelp() {
  process.stdout.write(`rednote interact

Usage:
  npx -y @skills-store/rednote interact [--instance NAME] --url URL [--like] [--collect] [--comment TEXT]
  node --experimental-strip-types ./scripts/rednote/interact.ts --instance NAME --url URL [--like] [--collect] [--comment TEXT]
  bun ./scripts/rednote/interact.ts --instance NAME --url URL [--like] [--collect] [--comment TEXT]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --url URL         Required. Xiaohongshu explore url
  --like            Optional. Perform like
  --collect         Optional. Perform collect
  --comment TEXT    Optional. Post comment content
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
      like: { type: 'boolean' },
      collect: { type: 'boolean' },
      comment: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  return {
    instance: values.instance,
    url: values.url,
    like: values.like,
    collect: values.collect,
    comment: values.comment,
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

function resolveInteractActions(values: InteractCliValues): { actions: InteractAction[]; commentContent?: string } {
  const actions: InteractAction[] = [];

  if (values.like) {
    actions.push('like');
  }

  if (values.collect) {
    actions.push('collect');
  }

  const commentContent = values.comment?.trim();
  if (commentContent) {
    actions.push('comment');
  }

  if (actions.length === 0) {
    throw new Error('At least one interact option is required: --like, --collect, or --comment');
  }

  return {
    actions,
    commentContent,
  };
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

async function typeCommentContent(page: Page, content: string) {
  const commentInput = page.locator(COMMENT_INPUT_SELECTOR);
  const visibleCommentInput = await requireVisibleLocator(
    commentInput,
    'Could not find the comment input. Make sure the feed detail page finished loading.',
    15_000,
  );

  await visibleCommentInput.scrollIntoViewIfNeeded();
  await visibleCommentInput.click({ force: true });
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A').catch(() => {});
  await page.keyboard.press('Backspace').catch(() => {});
  await page.keyboard.insertText(content);

  await page.waitForFunction(
    ({ selector, expectedContent }) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        return false;
      }

      return element.innerText.trim() === expectedContent;
    },
    { selector: COMMENT_INPUT_SELECTOR, expectedContent: content },
    { timeout: 5_000 },
  );
}

async function clickSendComment(page: Page) {
  const sendButton = page.locator(COMMENT_SEND_BUTTON_SELECTOR).filter({ hasText: COMMENT_SEND_BUTTON_TEXT });
  const visibleSendButton = await requireVisibleLocator(
    sendButton,
    'Could not find the Send button. Make sure the comment toolbar finished loading.',
    15_000,
  );

  await page.waitForFunction(
    ({ selector, text }) => {
      const buttons = [...document.querySelectorAll(selector)];
      const target = buttons.find((candidate) => candidate.textContent?.includes(text));
      return target instanceof HTMLButtonElement && !target.disabled;
    },
    { selector: COMMENT_SEND_BUTTON_SELECTOR, text: COMMENT_SEND_BUTTON_TEXT },
    { timeout: 5_000 },
  );

  await visibleSendButton.click();

  await page.waitForFunction(
    ({ inputSelector, buttonSelector, text }) => {
      const input = document.querySelector(inputSelector);
      const buttons = [...document.querySelectorAll(buttonSelector)];
      const button = buttons.find((candidate) => candidate.textContent?.includes(text));
      const inputCleared = input instanceof HTMLElement ? input.innerText.trim().length === 0 : false;
      const buttonDisabled = button instanceof HTMLButtonElement ? button.disabled : false;
      return inputCleared || buttonDisabled;
    },
    {
      inputSelector: COMMENT_INPUT_SELECTOR,
      buttonSelector: COMMENT_SEND_BUTTON_SELECTOR,
      text: COMMENT_SEND_BUTTON_TEXT,
    },
    { timeout: 10_000 },
  );
}

async function commentOnCurrentFeedPage(page: Page, content: string) {
  await typeCommentContent(page, content);
  await clickSendComment(page);
}

async function waitForInteractContainer(page: Page) {
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(500);
  await requireVisibleLocator(
    page.locator(INTERACT_CONTAINER_SELECTOR),
    'Could not find the interaction toolbar. Make sure the feed detail page finished loading.',
    15_000,
  );
}

function getActionErrorMessage(action: 'like' | 'collect') {
  return action === 'like'
    ? 'Could not find the Like button. Make sure the feed detail page finished loading.'
    : 'Could not find the Collect button. Make sure the feed detail page finished loading.';
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
  actions: InteractAction[],
  commentContent?: string,
): Promise<InteractResult> {
  validateFeedDetailUrl(url);
  const detailResult = await getFeedDetails(session, [url]);
  const detailItem = detailResult.detail.items[0];
  if (!detailItem) {
    throw new Error(`Failed to load feed detail: ${url}`);
  }

  const page = await getOrCreateXiaohongshuPage(session);
  await waitForInteractContainer(page);

  let liked = detailItem.note.interactInfo.liked === true;
  let collected = detailItem.note.interactInfo.collected === true;
  const messages: string[] = [];

  for (const action of actions) {
    if (action === 'like') {
      const alreadyActive = liked;
      await ensureActionApplied(page, 'like', alreadyActive);
      liked = true;
      messages.push(alreadyActive ? 'Like already active' : 'Like completed');
      continue;
    }

    if (action === 'collect') {
      const alreadyActive = collected;
      await ensureActionApplied(page, 'collect', alreadyActive);
      collected = true;
      messages.push(alreadyActive ? 'Collect already active' : 'Collect completed');
      continue;
    }

    const normalizedContent = ensureNonEmpty(commentContent, '--comment');
    await commentOnCurrentFeedPage(page, normalizedContent);
    messages.push('Comment posted');
  }

  return {
    ok: true,
    message: `${messages.join('; ')}: ${url}`,
  };
}

export async function runInteractCommand(values: InteractCliValues = {}) {
  if (values.help) {
    printInteractHelp();
    return;
  }

  const url = ensureNonEmpty(values.url, '--url');
  const { actions, commentContent } = resolveInteractActions(values);
  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, `performing ${actions.join(', ')} interact`, session);
    const result = await interactWithFeed(session, url, actions, commentContent);
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
