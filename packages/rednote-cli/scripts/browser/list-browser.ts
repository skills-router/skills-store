#!/usr/bin/env -S node --experimental-strip-types

import {
  browserSpecs,
  findSpec,
  inspectBrowserInstance,
  isLastConnectMatch,
  readInstanceStore,
  toBrowserInstanceInfo,
  type ListBrowserInstancesResult,
  type ListedBrowserInstance,
} from '../utils/browser-core.ts';
import {
  parseBrowserCliArgs,
  printJson,
  printListBrowserHelp,
  runCli,
  type BrowserCliValues,
} from '../utils/browser-cli.ts';

export async function listBrowserInstances(): Promise<ListBrowserInstancesResult> {
  const specs = browserSpecs();
  const store = readInstanceStore();
  const defaultInstances = await Promise.all(
    specs.map(async (spec) => {
      const instance = await inspectBrowserInstance(spec);
      return {
        ...toBrowserInstanceInfo(instance),
        scope: 'default',
        instanceName: spec.type,
        createdAt: null,
        lastConnect: isLastConnectMatch(store.lastConnect, 'default', spec.type, spec.type),
      } satisfies ListedBrowserInstance;
    }),
  );
  const customInstances = await Promise.all(
    store.instances.map(async (persisted) => {
      const spec = findSpec(persisted.browser);
      const instance = await inspectBrowserInstance(spec, undefined, persisted.userDataDir);
      return {
        ...toBrowserInstanceInfo(instance),
        name: persisted.name,
        userDataDir: persisted.userDataDir,
        scope: 'custom',
        instanceName: persisted.name,
        createdAt: persisted.createdAt,
        lastConnect: isLastConnectMatch(store.lastConnect, 'custom', persisted.name, persisted.browser),
      } satisfies ListedBrowserInstance;
    }),
  );

  return {
    lastConnect: store.lastConnect,
    instances: [...defaultInstances, ...customInstances],
  };
}

export async function runListBrowserCommand(_values?: BrowserCliValues) {
  const instances = await listBrowserInstances();
  printJson(instances);
}

async function main() {
  const { values } = parseBrowserCliArgs(process.argv.slice(2));

  if (values.help) {
    printListBrowserHelp();
    return;
  }

  await runListBrowserCommand(values);
}

runCli(import.meta.url, main);
