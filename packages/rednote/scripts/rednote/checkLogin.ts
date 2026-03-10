#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import type { Browser, BrowserContext, Page } from 'playwright-core';
import { initBrowser, resolveConnectOptions } from '../browser/connect-browser.ts';
import { connectOverCdp, updateLastConnect } from '../utils/browser-core.ts';
import { debugLog, printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget, type RednoteAccountStatus, type RednoteStatusTarget } from './status.ts';

export type CheckLoginCliValues = {
  instance?: string;
  help?: boolean;
};

export type CheckLoginResult = {
  ok: true;
  rednote: RednoteAccountStatus & {
    needLogin: boolean;
    checkedAt: string;
  };
};

export type RednoteSession = {
  browser: Browser;
  browserContext: BrowserContext;
  page: Page;
};

function printCheckLoginHelp() {
  process.stdout.write(`rednote check-login

Usage:
  npx -y @skills-store/rednote check-login [--instance NAME]
  node --experimental-strip-types ./scripts/rednote/checkLogin.ts --instance NAME
  bun ./scripts/rednote/checkLogin.ts --instance NAME

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  -h, --help        Show this help
`);
}

export async function createRednoteSession(target: RednoteStatusTarget): Promise<RednoteSession> {
  debugLog('checkLogin', 'create session start', { target });
  const resolved = await resolveConnectOptions(
    target.scope === 'custom'
      ? { instanceName: target.instanceName }
      : { browser: target.browser },
  );
  const launched = await initBrowser(resolved.connectOptions);
  debugLog('checkLogin', 'initBrowser resolved', {
    target,
    connectOptions: resolved.connectOptions,
    launched,
  });

  if (resolved.lastConnect) {
    updateLastConnect(resolved.lastConnect);
  }

  const browser = await connectOverCdp(launched.wsUrl);
  debugLog('checkLogin', 'connected over cdp', { wsUrl: launched.wsUrl, remoteDebuggingPort: launched.remoteDebuggingPort });
  const browserContext = browser.contexts()[0];

  if (!browserContext) {
    throw new Error(`No browser context found for instance: ${target.instanceName}`);
  }

  const page = browserContext.pages().find((candidate) => candidate.url().startsWith('https://www.xiaohongshu.com/')) ?? await browserContext.newPage();
  debugLog('checkLogin', 'session page resolved', { pageUrl: page.url(), totalPages: browserContext.pages().length });

  return {
    browser,
    browserContext,
    page,
  };
}

export async function disconnectRednoteSession(session: RednoteSession) {
  try {
    await session.browser.close();
  } catch {
  }
}

export async function checkRednoteLogin(
  target: RednoteStatusTarget,
  session?: RednoteSession,
): Promise<RednoteAccountStatus & {
  needLogin: boolean;
  checkedAt: string;
}> {
  const ownsSession = !session;
  const activeSession = session ?? await createRednoteSession(target);

  try {
    debugLog('checkLogin', 'check login start', { target, reusedSession: Boolean(session) });
    const page = activeSession.page;

    if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
      debugLog('checkLogin', 'page is not on xiaohongshu, navigating', { currentUrl: page.url() });
      await page.goto('https://www.xiaohongshu.com/explore', {
        waitUntil: 'domcontentloaded',
      });
    }

    await page.waitForTimeout(2_000);
    const needLogin = (await page.locator('#login-btn').count()) > 0;
    debugLog('checkLogin', 'login state checked', { pageUrl: page.url(), needLogin });

    return {
      loginStatus: needLogin ? 'logged-out' : 'logged-in',
      lastLoginAt: null,
      needLogin,
      checkedAt: new Date().toISOString(),
    };
  } finally {
    if (ownsSession) {
      await disconnectRednoteSession(activeSession);
    }
  }
}

export async function ensureRednoteLoggedIn(target: RednoteStatusTarget, action = 'continue', session?: RednoteSession) {
  const rednote = await checkRednoteLogin(target, session);

  if (rednote.needLogin) {
    throw new Error(`Xiaohongshu login is required before ${action}. Run \`rednote login --instance ${target.instanceName}\` first.`);
  }

  return rednote;
}

export async function runCheckLoginCommand(values: CheckLoginCliValues = {}) {
  if (values.help) {
    printCheckLoginHelp();
    return;
  }


  const target = resolveStatusTarget(values.instance);
  const rednote = await checkRednoteLogin(target);

  const result: CheckLoginResult = {
    ok: true,
    rednote,
  };

  printJson(result);
}

async function main() {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help) {
    printCheckLoginHelp();
    return;
  }

  await runCheckLoginCommand(values);
}

runCli(import.meta.url, main);
