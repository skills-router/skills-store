#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import {
  findSpec,
  inspectBrowserInstance,
  isDefaultInstanceName,
  isLastConnectMatch,
  readInstanceStore,
  type BrowserInstanceInfo,
  type BrowserType,
  type PersistedInstance,
} from '../utils/browser-core.ts';
import { debugLog, printJson, runCli, stringifyError } from '../utils/browser-cli.ts';

export type RednoteLoginStatus = 'logged-in' | 'logged-out' | 'unknown';

export type RednoteAccountStatus = {
  loginStatus: RednoteLoginStatus;
  lastLoginAt: string | null;
};

export type RednoteStatusSource = 'argument' | 'last-connect' | 'single-instance';

export type RednoteStatusTarget = {
  scope: 'default' | 'custom';
  instanceName: string;
  browser: BrowserType;
  userDataDir: string | null;
  createdAt: string | null;
  lastConnect: boolean;
  source: RednoteStatusSource;
};

export type ResolveRednoteAccountStatusContext = RednoteStatusTarget;

export type RednoteAccountStatusProvider = (
  context: ResolveRednoteAccountStatusContext,
) => Promise<RednoteAccountStatus> | RednoteAccountStatus;

export type RednoteInstanceState = 'running' | 'stopped' | 'missing' | 'stale-lock';

export type RednoteStatusResult = {
  ok: true;
  instance: {
    scope: 'default' | 'custom';
    name: string;
    browser: BrowserType;
    source: RednoteStatusSource;
    status: RednoteInstanceState;
    exists: boolean;
    inUse: boolean;
    pid: number | null;
    remotePort: number | null;
    userDataDir: string;
    createdAt: string | null;
    lastConnect: boolean;
  };
  rednote: RednoteAccountStatus;
};

export type StatusCliValues = {
  instance?: string;
  help?: boolean;
};

let rednoteAccountStatusProvider: RednoteAccountStatusProvider = async () => ({
  loginStatus: 'unknown',
  lastLoginAt: null,
});

export function registerRednoteAccountStatusProvider(provider: RednoteAccountStatusProvider) {
  rednoteAccountStatusProvider = provider;
}

export async function getRednoteAccountStatus(context: ResolveRednoteAccountStatusContext) {
  return await rednoteAccountStatusProvider(context);
}

function printStatusHelp() {
  process.stdout.write(`rednote status

Usage:
  npx -y @skills-store/rednote status [--instance NAME]
  node --experimental-strip-types ./scripts/rednote/status.ts [--instance NAME]
  bun ./scripts/rednote/status.ts [--instance NAME]

Options:
  --instance NAME   Show status for a custom instance or default browser instance
  -h, --help        Show this help
`);
}

function toInstanceState(instance: BrowserInstanceInfo): RednoteInstanceState {
  if (instance.staleLock) {
    return 'stale-lock';
  }

  if (instance.inUse) {
    return 'running';
  }

  if (instance.exists) {
    return 'stopped';
  }

  return 'missing';
}

function fromPersistedInstance(
  instance: PersistedInstance,
  source: RednoteStatusSource,
): RednoteStatusTarget {
  const store = readInstanceStore();

  return {
    scope: 'custom',
    instanceName: instance.name,
    browser: instance.browser,
    userDataDir: instance.userDataDir,
    createdAt: instance.createdAt,
    lastConnect: isLastConnectMatch(store.lastConnect, 'custom', instance.name, instance.browser),
    source,
  };
}

export function resolveStatusTarget(instanceName?: string): RednoteStatusTarget {
  const store = readInstanceStore();

  if (instanceName) {
    const normalizedName = instanceName.trim();
    if (!normalizedName) {
      throw new Error('Instance name cannot be empty');
    }

    if (isDefaultInstanceName(normalizedName)) {
      const browser = normalizedName as BrowserType;
      return {
        scope: 'default',
        instanceName: browser,
        browser,
        userDataDir: null,
        createdAt: null,
        lastConnect: isLastConnectMatch(store.lastConnect, 'default', browser, browser),
        source: 'argument',
      };
    }

    const persisted = store.instances.find((item) => item.name === normalizedName);
    if (!persisted) {
      throw new Error(`Unknown instance: ${normalizedName}`);
    }

    return {
      scope: 'custom',
      instanceName: persisted.name,
      browser: persisted.browser,
      userDataDir: persisted.userDataDir,
      createdAt: persisted.createdAt,
      lastConnect: isLastConnectMatch(store.lastConnect, 'custom', persisted.name, persisted.browser),
      source: 'argument',
    };
  }

  if (store.lastConnect) {
    if (store.lastConnect.scope === 'default') {
      return {
        scope: 'default',
        instanceName: store.lastConnect.name,
        browser: store.lastConnect.browser,
        userDataDir: null,
        createdAt: null,
        lastConnect: true,
        source: 'last-connect',
      };
    }

    const persisted = store.instances.find((item) => item.name === store.lastConnect?.name);
    if (persisted) {
      return {
        scope: 'custom',
        instanceName: persisted.name,
        browser: persisted.browser,
        userDataDir: persisted.userDataDir,
        createdAt: persisted.createdAt,
        lastConnect: true,
        source: 'last-connect',
      };
    }
  }

  if (store.instances.length === 1) {
    return fromPersistedInstance(store.instances[0], 'single-instance');
  }

  throw new Error('No current instance found. Use --instance NAME or connect an instance first.');
}

export async function getRednoteStatus(target: RednoteStatusTarget): Promise<RednoteStatusResult> {
  debugLog('status', 'get status start', { target });
  const spec = findSpec(target.browser);
  const inspected = await inspectBrowserInstance(
    spec,
    undefined,
    target.scope === 'custom' ? target.userDataDir ?? undefined : undefined,
  );
  debugLog('status', 'instance inspected', { inspected });

  let rednote = await getRednoteAccountStatus(target);
  debugLog('status', 'account status provider result', { rednote });
  if (rednote.loginStatus === 'unknown') {
    try {
      debugLog('status', 'login status unknown, fallback to checkRednoteLogin', { target });
      const { checkRednoteLogin } = await import('./checkLogin.ts');
      const checked = await checkRednoteLogin(target);
      debugLog('status', 'fallback checkRednoteLogin succeeded', { checked });
      rednote = {
        loginStatus: checked.loginStatus,
        lastLoginAt: checked.lastLoginAt,
      };
    } catch (error) {
      debugLog('status', 'fallback checkRednoteLogin failed', { error: stringifyError(error) });
      rednote = {
        loginStatus: 'unknown',
        lastLoginAt: null,
      };
    }
  }

  return {
    ok: true,
    instance: {
      scope: target.scope,
      name: target.instanceName,
      browser: target.browser,
      source: target.source,
      status: toInstanceState(inspected),
      exists: inspected.exists,
      inUse: inspected.inUse,
      pid: inspected.pid,
      remotePort: inspected.remotePort,
      userDataDir: inspected.userDataDir,
      createdAt: target.createdAt,
      lastConnect: target.lastConnect,
    },
    rednote,
  };
}

export async function runStatusCommand(values: StatusCliValues = {}) {
  if (values.help) {
    printStatusHelp();
    return;
  }

  const target = resolveStatusTarget(values.instance);
  const result = await getRednoteStatus(target);
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
    printStatusHelp();
    return;
  }

  await runStatusCommand(values);
}

runCli(import.meta.url, main);
