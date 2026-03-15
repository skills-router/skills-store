#!/usr/bin/env -S node --experimental-strip-types

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import type { Page } from 'playwright-core';
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
    qrCodePath: string | null;
    message: string;
  };
};

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REDNOTE_ROOT = path.resolve(SCRIPT_DIR, '../..');

function timestampForFilename() {
  return new Date().toISOString().replaceAll(':', '').replaceAll('.', '').replace('T', '-').replace('Z', 'Z');
}

function resolveQrCodePath() {
  return path.join(REDNOTE_ROOT, 'output', `login-qrcode-${timestampForFilename()}.png`);
}

function parseQrCodeDataUrl(src: string) {
  const match = src.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) {
    return null;
  }

  return {
    mimeType: match[1],
    buffer: Buffer.from(match[2], 'base64'),
  };
}

async function refreshExpiredQrCode(page: Page) {
  const statusText = page.locator('.qrcode .status-text').first();
  const refreshButton = page.locator('.qrcode .status-desc.refresh').first();
  const isExpiredVisible = await statusText.isVisible().catch(() => false);

  if (!isExpiredVisible) {
    return false;
  }

  const text = (await statusText.textContent().catch(() => null))?.trim() ?? '';

  if (!text.includes('过期')) {
    return false;
  }

  if (await refreshButton.isVisible().catch(() => false)) {
    await refreshButton.click({ timeout: 2_000 }).catch(() => {});
    await page.waitForTimeout(800);
    return true;
  }

  return false;
}

async function saveQrCodeImage(page: Page) {
  const qrImage = page.locator('.qrcode .qrcode-img').first();

  for (let attempt = 0; attempt < 3; attempt += 1) {
    await qrImage.waitFor({ state: 'visible', timeout: 5_000 });

    const refreshed = await refreshExpiredQrCode(page);
    if (refreshed) {
      continue;
    }

    const filePath = resolveQrCodePath();
    fs.mkdirSync(path.dirname(filePath), { recursive: true });

    const src = await qrImage.getAttribute('src');
    if (src) {
      const parsed = parseQrCodeDataUrl(src);
      if (parsed?.mimeType === 'image/png') {
        fs.writeFileSync(filePath, parsed.buffer);
        return filePath;
      }
    }

    await qrImage.screenshot({ path: filePath });
    return filePath;
  }

  throw new Error('No usable Xiaohongshu login QR code was detected. Make sure the login dialog is open.');
}

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
        qrCodePath: null,
        message: 'The current instance is already logged in. No additional login step is required.',
      },
    };
  }

  const { page } = await getOrCreateXiaohongshuPage(session);
  await page.reload();
  const loginButton = page.locator('#login-btn');
  const hasLoginButton = (await loginButton.count()) > 0;

  if (!hasLoginButton) {
    return {
      ok: true,
      rednote: {
        loginClicked: false,
        pageUrl: page.url(),
        waitingForPhoneLogin: false,
        qrCodePath: null,
        message: 'No login button was found. The current instance may already be logged in.',
      },
    };
  }
  await loginButton.first().click({ timeout: 2000, force: true });
  await page.waitForTimeout(500);
  const qrCodePath = await saveQrCodeImage(page);

  return {
    ok: true,
    rednote: {
      loginClicked: true,
      pageUrl: page.url(),
      waitingForPhoneLogin: true,
      qrCodePath,
      message: 'The login button was clicked and the QR code image was exported. Scan the code to finish logging in.',
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
    await disconnectRednoteSession(session);
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
