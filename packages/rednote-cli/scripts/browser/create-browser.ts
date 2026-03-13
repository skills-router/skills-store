#!/usr/bin/env -S node --experimental-strip-types

import {
  customInstanceUserDataDir,
  ensureDir,
  isDefaultInstanceName,
  readInstanceStore,
  type CreateInstanceOptions,
  type CreateInstanceResult,
  type PersistedInstance,
  validateInstanceName,
  writeInstanceStore,
} from '../utils/browser-core.ts';
import {
  parseBrowserCliArgs,
  printCreateBrowserHelp,
  printJson,
  runCli,
  type BrowserCliValues,
} from '../utils/browser-cli.ts';

export function createBrowserInstance(options: CreateInstanceOptions): CreateInstanceResult {
  const store = readInstanceStore();
  const name = validateInstanceName(options.name);
  const browser = options.browser ?? 'chrome';

  if (isDefaultInstanceName(name)) {
    throw new Error(`Instance name is reserved for a default browser: ${name}`);
  }

  if (store.instances.some((instance) => instance.name === name)) {
    throw new Error(`Instance already exists: ${name}`);
  }

  const instance: PersistedInstance = {
    name,
    browser,
    userDataDir: ensureDir(customInstanceUserDataDir(name)),
    createdAt: new Date().toISOString(),
    remoteDebuggingPort:
      typeof options.remoteDebuggingPort === 'number' && Number.isInteger(options.remoteDebuggingPort) && options.remoteDebuggingPort > 0
        ? options.remoteDebuggingPort
        : undefined,
  };

  writeInstanceStore({
    ...store,
    instances: [...store.instances, instance],
  });

  return {
    ok: true,
    instance,
  };
}

export function runCreateBrowserCommand(values: BrowserCliValues) {
  if (!values.name) {
    throw new Error('Missing required option: --name');
  }

  const result = createBrowserInstance({
    name: values.name,
    browser: values.browser,
    remoteDebuggingPort: values.port ? Number(values.port) : undefined,
  });
  printJson(result);
}

async function main() {
  const { values } = parseBrowserCliArgs(process.argv.slice(2));

  if (values.help) {
    printCreateBrowserHelp();
    return;
  }

  runCreateBrowserCommand(values);
}

runCli(import.meta.url, main);
