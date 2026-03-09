import path from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';

export type BrowserCliValues = {
  browser?: 'chrome' | 'edge' | 'chromium' | 'brave';
  instance?: string;
  name?: string;
  'executable-path'?: string;
  'user-data-dir'?: string;
  force?: boolean;
  port?: string;
  timeout?: string;
  'kill-timeout'?: string;
  'startup-url'?: string;
  help?: boolean;
};

export type BrowserCliArgs = {
  values: BrowserCliValues;
  positionals: string[];
};

function printHelp(helpText: string) {
  process.stdout.write(helpText);
}

export function parseBrowserCliArgs(argv: string[]): BrowserCliArgs {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      browser: { type: 'string' },
      instance: { type: 'string' },
      name: { type: 'string' },
      'executable-path': { type: 'string' },
      'user-data-dir': { type: 'string' },
      force: { type: 'boolean' },
      port: { type: 'string' },
      timeout: { type: 'string' },
      'kill-timeout': { type: 'string' },
      'startup-url': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  return {
    values: values as BrowserCliValues,
    positionals,
  };
}

export function printJson(value: unknown) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

export function printInitBrowserHelp() {
  printHelp(`rednote browser

Commands:
  list
  create --name NAME [--browser chrome|edge|chromium|brave] [--port 9222]
  remove --name NAME [--force]
  connect [--instance NAME] [--browser chrome|edge|chromium|brave] [--user-data-dir PATH] [--force] [--port 9222]

Examples:
  npx -y @skills-store/rednote browser list
  npx -y @skills-store/rednote browser create --name seller-main --browser chrome --port 9222
  npx -y @skills-store/rednote browser remove --name seller-main
  npx -y @skills-store/rednote browser connect --instance seller-main
  npx -y @skills-store/rednote browser connect --browser edge --user-data-dir /tmp/edge-profile --port 9223
`);
}

export function printCreateBrowserHelp() {
  printHelp(`rednote browser create

Usage:
  npx -y @skills-store/rednote browser create --name NAME [--browser chrome|edge|chromium|brave] [--port 9222]
  bun ./scripts/browser/create-browser.ts --name NAME [--browser chrome|edge|chromium|brave] [--port 9222]
`);
}

export function printListBrowserHelp() {
  printHelp(`rednote browser list

Usage:
  npx -y @skills-store/rednote browser list
  bun ./scripts/browser/list-browser.ts
`);
}

export function printRemoveBrowserHelp() {
  printHelp(`rednote browser remove

Usage:
  npx -y @skills-store/rednote browser remove --name NAME [--force]
  bun ./scripts/browser/remove-browser.ts --name NAME [--force]
`);
}

export function printConnectBrowserHelp() {
  printHelp(`rednote browser connect

Usage:
  npx -y @skills-store/rednote browser connect [--instance NAME] [--browser chrome|edge|chromium|brave] [--user-data-dir PATH] [--force] [--port 9222]
  bun ./scripts/browser/connect-browser.ts [--instance NAME] [--browser chrome|edge|chromium|brave] [--user-data-dir PATH] [--force] [--port 9222]

Notes:
  When using --instance without --port, the stored instance port from data.json is used.
  If no stored port exists yet, a random free port is assigned and saved for next time.
`);
}

export function stringifyError(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function isDebugEnabled() {
  const value = process.env.REDNOTE_DEBUG?.trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

export function debugLog(scope: string, message: string, payload?: Record<string, unknown>) {
  if (!isDebugEnabled()) {
    return;
  }

  const time = new Date().toISOString();
  const suffix = payload ? ` ${JSON.stringify(payload)}` : '';
  process.stderr.write(`[rednote-debug][${time}][${scope}] ${message}${suffix}
`);
}

export function isMainModule(metaUrl: string) {
  const entryArg = process.argv[1];
  if (!entryArg) {
    return false;
  }
  return metaUrl === pathToFileURL(path.resolve(entryArg)).href;
}

async function finalizeCliProcess(exitCode: number) {
  await new Promise<void>((resolve) => process.stdout.write('', () => resolve()));
  await new Promise<void>((resolve) => process.stderr.write('', () => resolve()));
  process.exit(exitCode);
}

export function runCli(metaUrl: string, main: () => Promise<void>) {
  if (!isMainModule(metaUrl)) {
    return;
  }

  main()
    .then(async () => {
      await finalizeCliProcess(0);
    })
    .catch(async (error) => {
      process.stderr.write(
        `${JSON.stringify(
          {
            ok: false,
            error: stringifyError(error),
          },
          null,
          2,
        )}
`,
      );
      await finalizeCliProcess(1);
    });
}
