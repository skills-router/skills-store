#!/usr/bin/env -S node --experimental-strip-types

import fs from 'node:fs';
import {
  closeProcessesByPid,
  customInstanceUserDataDir,
  exists,
  findSpec,
  inspectBrowserInstance,
  INSTANCES_DIR,
  isDefaultInstanceName,
  isSubPath,
  normalizePath,
  readInstanceStore,
  type RemoveInstanceOptions,
  type RemoveInstanceResult,
  validateInstanceName,
  waitForPortToClose,
  writeInstanceStore,
} from '../utils/browser-core.ts';
import {
  parseBrowserCliArgs,
  printJson,
  printRemoveBrowserHelp,
  runCli,
  type BrowserCliValues,
} from '../utils/browser-cli.ts';

export async function removeBrowserInstance(options: RemoveInstanceOptions): Promise<RemoveInstanceResult> {
  const store = readInstanceStore();
  const name = validateInstanceName(options.name);
  if (isDefaultInstanceName(name)) {
    throw new Error(`Default instance cannot be removed: ${name}`);
  }

  const instance = store.instances.find((item) => item.name === name);
  if (!instance) {
    throw new Error(`Instance not found: ${name}`);
  }

  const expectedUserDataDir = customInstanceUserDataDir(name);
  const resolvedUserDataDir = normalizePath(instance.userDataDir);
  if (!isSubPath(INSTANCES_DIR, resolvedUserDataDir) || normalizePath(expectedUserDataDir) !== resolvedUserDataDir) {
    throw new Error(`Refusing to remove unexpected instance directory: ${instance.userDataDir}`);
  }

  const spec = findSpec(instance.browser);
  const detectedInstance = await inspectBrowserInstance(spec, undefined, resolvedUserDataDir);
  if (detectedInstance.inUse && !options.force) {
    throw new Error(`Instance is currently in use. Re-run with --force: ${name}`);
  }

  const closedPids: number[] = [];
  if (options.force && detectedInstance.pids.length > 0) {
    for (const pid of await closeProcessesByPid(detectedInstance.pids, 8_000)) {
      closedPids.push(pid);
    }

    if (detectedInstance.remotePort !== null) {
      await waitForPortToClose(detectedInstance.remotePort, 8_000);
    }
  }

  const removedDir = exists(resolvedUserDataDir);
  fs.rmSync(resolvedUserDataDir, { recursive: true, force: true });

  writeInstanceStore({
    ...store,
    lastConnect:
      store.lastConnect?.scope === 'custom' &&
      store.lastConnect.name === instance.name &&
      store.lastConnect.browser === instance.browser
        ? null
        : store.lastConnect,
    instances: store.instances.filter((item) => item.name !== instance.name),
  });

  return {
    ok: true,
    removedInstance: instance,
    removedDir,
    closedPids,
  };
}

export async function runRemoveBrowserCommand(values: BrowserCliValues) {
  if (!values.name) {
    throw new Error('Missing required option: --name');
  }

  const result = await removeBrowserInstance({
    name: values.name,
    force: values.force,
  });
  printJson(result);
}

async function main() {
  const { values } = parseBrowserCliArgs(process.argv.slice(2));

  if (values.help) {
    process.stdout.write(`remove-browser\n\nUsage:\n  node --experimental-strip-types ./scripts/browser/remove-browser.ts --name NAME [--force]\n  bun ./scripts/browser/remove-browser.ts --name NAME [--force]\n`);
    return;
  }

  await runRemoveBrowserCommand(values);
}

runCli(import.meta.url, main);
