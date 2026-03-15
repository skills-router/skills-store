#!/usr/bin/env node
import { runRootCli } from '../dist/index.js';
import { stringifyError } from '../dist/utils/browser-cli.js';

async function finalizeCliProcess(exitCode) {
  await new Promise((resolve) => process.stdout.write('', () => resolve()));
  await new Promise((resolve) => process.stderr.write('', () => resolve()));
  process.exit(exitCode);
}

try {
  await runRootCli(process.argv.slice(2));
  await finalizeCliProcess(0);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        error: stringifyError(error),
      },
      null,
      2,
    )}\n`,
  );
  await finalizeCliProcess(1);
}
