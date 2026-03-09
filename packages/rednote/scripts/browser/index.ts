#!/usr/bin/env -S node --experimental-strip-types

import {
  parseBrowserCliArgs,
  printInitBrowserHelp,
  runCli,
} from '../utils/browser-cli.ts';

export async function runBrowserCli(argv: string[] = process.argv.slice(2)) {
  const { values, positionals } = parseBrowserCliArgs(argv);
  const command = positionals[0] || 'list';

  if (values.help || command === 'help') {
    printInitBrowserHelp();
    return;
  }

  if (command === 'list') {
    const { runListBrowserCommand } = await import('./list-browser.ts');
    await runListBrowserCommand(values);
    return;
  }

  if (command === 'create') {
    const { runCreateBrowserCommand } = await import('./create-browser.ts');
    runCreateBrowserCommand(values);
    return;
  }

  if (command === 'remove') {
    const { runRemoveBrowserCommand } = await import('./remove-browser.ts');
    await runRemoveBrowserCommand(values);
    return;
  }

  if (command === 'connect') {
    const { runConnectBrowserCommand } = await import('./connect-browser.ts');
    await runConnectBrowserCommand(values);
    return;
  }

  throw new Error(`Unknown browser command: ${command}`);
}

async function main() {
  await runBrowserCli();
}

runCli(import.meta.url, main);
