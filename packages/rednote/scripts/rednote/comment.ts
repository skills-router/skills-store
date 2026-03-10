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

const COMMENT_INPUT_SELECTOR = '#content-textarea[contenteditable="true"]';
const COMMENT_SEND_BUTTON_SELECTOR = 'button.btn.submit';
const COMMENT_SEND_BUTTON_TEXT = '发送';

export type CommentCliValues = {
  instance?: string;
  url?: string;
  content?: string;
  help?: boolean;
};

export type CommentResult = {
  ok: true;
  comment: {
    url: string;
    content: string;
    commentedAt: string;
  };
};

function printCommentHelp() {
  process.stdout.write(`rednote comment

Usage:
  npx -y @skills-store/rednote comment [--instance NAME] --url URL --content TEXT
  node --experimental-strip-types ./scripts/rednote/comment.ts --instance NAME --url URL --content TEXT
  bun ./scripts/rednote/comment.ts --instance NAME --url URL --content TEXT

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --url URL         Required. Xiaohongshu explore url
  --content TEXT    Required. Comment content to send
  -h, --help        Show this help
`);
}

export function parseCommentCliArgs(argv: string[]): CommentCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      url: { type: 'string' },
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
    '未找到评论输入框，请确认帖子详情页已正确加载。',
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
    '未找到“发送”按钮，请确认评论工具栏已正确加载。',
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

export async function commentOnFeed(session: RednoteSession, url: string, content: string): Promise<CommentResult> {
  validateFeedDetailUrl(url);
  const page = await getOrCreateXiaohongshuPage(session);

  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('domcontentloaded');
  await page.waitForTimeout(1_000);

  await typeCommentContent(page, content);
  await clickSendComment(page);

  return {
    ok: true,
    comment: {
      url,
      content,
      commentedAt: new Date().toISOString(),
    },
  };
}

export async function runCommentCommand(values: CommentCliValues = {}) {
  if (values.help) {
    printCommentHelp();
    return;
  }

  const url = ensureNonEmpty(values.url, '--url');
  const content = ensureNonEmpty(values.content, '--content');
  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'commenting on feed', session);
    const result = await commentOnFeed(session, url, content);
    printJson(result);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseCommentCliArgs(process.argv.slice(2));
  await runCommentCommand(values);
}

runCli(import.meta.url, main);
