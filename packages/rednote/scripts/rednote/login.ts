#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget, type RednoteStatusTarget } from './status.ts';
import { checkRednoteLogin, createRednoteSession, disconnectRednoteSession, type RednoteSession } from './checkLogin.ts';

export type LoginCliValues = {
  instance?: string;
  help?: boolean;
};

export type LoginResult = {
  ok: true;
  rednote: {
    loginClicked: boolean;
    pageUrl: string;
    waitingForPhoneLogin: boolean;
    message: string;
  };
};

function printLoginHelp() {
  process.stdout.write(`rednote login

Usage:
  npx -y @skills-store/rednote login [--instance NAME]
  node --experimental-strip-types ./scripts/rednote/login.ts --instance NAME
  bun ./scripts/rednote/login.ts --instance NAME

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  -h, --help        Show this help
`);
}

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  const page = session.page;

  if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
    await page.goto('https://www.xiaohongshu.com/explore', {
      waitUntil: 'domcontentloaded',
    });
  }

  return { page };
}

export async function openRednoteLogin(target: RednoteStatusTarget, session: RednoteSession): Promise<LoginResult> {
  const rednoteStatus = await checkRednoteLogin(target, session);

  if (!rednoteStatus.needLogin) {
    return {
      ok: true,
      rednote: {
        loginClicked: false,
        pageUrl: session.page.url(),
        waitingForPhoneLogin: false,
        message: '当前实例已登录，无需重复执行登录操作。',
      },
    };
  }

  const { page } = await getOrCreateXiaohongshuPage(session);
  const loginButton = page.locator('#login-btn');
  const hasLoginButton = (await loginButton.count()) > 0;

  if (!hasLoginButton) {
    return {
      ok: true,
      rednote: {
        loginClicked: false,
        pageUrl: page.url(),
        waitingForPhoneLogin: false,
        message: '未检测到登录按钮，当前实例可能已经登录。',
      },
    };
  }
  await loginButton.first().click({ timeout: 2000 });
  await page.waitForTimeout(500);

  return {
    ok: true,
    rednote: {
      loginClicked: true,
      pageUrl: page.url(),
      waitingForPhoneLogin: true,
      message: '已点击登录按钮，请在浏览器中继续输入手机号并完成登录。',
    },
  };
}

export async function runLoginCommand(values: LoginCliValues = {}) {
  if (values.help) {
    printLoginHelp();
    return;
  }


  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    const result = await openRednoteLogin(target, session);
    printJson(result);
  } finally {
    disconnectRednoteSession(session);
  }
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
    printLoginHelp();
    return;
  }

  await runLoginCommand(values);
}

runCli(import.meta.url, main);
