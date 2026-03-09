#!/usr/bin/env -S node --experimental-strip-types

import {
  closeProcessesByPid,
  findSpec,
  getCdpWebSocketUrl,
  getPidsListeningOnPort,
  inspectBrowserInstance,
  getRandomAvailablePort,
  launchBrowser,
  removeLockFiles,
  readInstanceStore,
  resolveExecutablePath,
  resolveUserDataDir,
  type ConnectBrowserOptions,
  type ConnectBrowserResult,
  type LastConnectInfo,
  updateInstanceRemoteDebuggingPort,
  updateLastConnect,
  validateInstanceName,
  waitForCdpReady,
  waitForPortToClose,
} from '../utils/browser-core.ts';
import {
  parseBrowserCliArgs,
  printConnectBrowserHelp,
  printJson,
  runCli,
  debugLog,
  type BrowserCliValues,
} from '../utils/browser-cli.ts';


async function resolveBrowserPid(remoteDebuggingPort: number, detectedPid: number | null) {
  if (detectedPid) {
    return detectedPid;
  }

  const portPids = await getPidsListeningOnPort(remoteDebuggingPort);
  return portPids[0] ?? null;
}

export async function resolveConnectOptions(options: ConnectBrowserOptions & { instanceName?: string }) {
  if (!options.instanceName) {
    return {
      connectOptions: options,
      lastConnect:
        options.userDataDir
          ? null
          : {
              scope: 'default',
              name: options.browser ?? 'chrome',
              browser: options.browser ?? 'chrome',
            } satisfies LastConnectInfo,
    };
  }

  if (options.userDataDir) {
    throw new Error('Do not combine --instance with --user-data-dir');
  }

  if (options.browser) {
    throw new Error('Do not combine --instance with --browser');
  }

  const store = readInstanceStore();
  const instanceName = validateInstanceName(options.instanceName);
  const persisted = store.instances.find((instance) => instance.name === instanceName);
  if (!persisted) {
    throw new Error(`Unknown instance: ${instanceName}`);
  }

  let remoteDebuggingPort = options.remoteDebuggingPort ?? persisted.remoteDebuggingPort;
  if (!remoteDebuggingPort) {
    remoteDebuggingPort = await getRandomAvailablePort();
    updateInstanceRemoteDebuggingPort(instanceName, remoteDebuggingPort);
  }

  return {
    connectOptions: {
      ...options,
      browser: persisted.browser,
      userDataDir: persisted.userDataDir,
      remoteDebuggingPort,
    } satisfies ConnectBrowserOptions,
    lastConnect: {
      scope: 'custom',
      name: persisted.name,
      browser: persisted.browser,
    } satisfies LastConnectInfo,
  };
}

export async function initBrowser(options: ConnectBrowserOptions = {}): Promise<ConnectBrowserResult> {
  debugLog('initBrowser', 'start', { options });
  const browserType = options.browser ?? 'chrome';
  const spec = findSpec(browserType);
  const executablePath = options.executablePath || resolveExecutablePath(spec);
  if (!executablePath) {
    throw new Error(`No executable found for ${spec.displayName}`);
  }

  const userDataDir = resolveUserDataDir(spec, options);
  const remoteDebuggingPort = options.remoteDebuggingPort ?? 9222;
  const connectTimeoutMs = options.connectTimeoutMs ?? 15_000;
  const killTimeoutMs = options.killTimeoutMs ?? 8_000;
  const startupUrl = options.startupUrl ?? 'about:blank';

  const detectedInstance = await inspectBrowserInstance(spec, executablePath, userDataDir);
  debugLog('initBrowser', 'inspected instance', {
    browserType,
    executablePath,
    userDataDir,
    remoteDebuggingPort,
    detectedInstance,
  });
  const closedPidSet = new Set<number>();

  const existingWsUrl = await getCdpWebSocketUrl(remoteDebuggingPort);
  debugLog('initBrowser', 'checked existing ws url', { remoteDebuggingPort, existingWsUrl });
  if (existingWsUrl) {
    debugLog('initBrowser', 'reusing existing ws url', { remoteDebuggingPort, existingWsUrl });
    return {
      ok: true,
      type: spec.type,
      executablePath,
      userDataDir,
      remoteDebuggingPort,
      wsUrl: existingWsUrl,
      pid: await resolveBrowserPid(remoteDebuggingPort, detectedInstance.pid),
    };
  }

  if (detectedInstance.inUse && detectedInstance.remotePort === null) {
    if (!options.force) {
      throw new Error(
        `Browser profile is already in use without a remote debugging port. Re-run with --force to safely close it first: ${detectedInstance.userDataDir}`,
      );
    }

    if (detectedInstance.pids.length === 0) {
      throw new Error(
        `Browser profile is in use but no running browser PID was found for safe shutdown: ${detectedInstance.userDataDir}`,
      );
    }
  }

  if (detectedInstance.inUse && detectedInstance.pids.length > 0) {
    debugLog('initBrowser', 'detected in-use profile, closing existing processes', {
      pids: detectedInstance.pids,
      killTimeoutMs,
      remoteDebuggingPort,
    });
    for (const pid of await closeProcessesByPid(detectedInstance.pids, killTimeoutMs)) {
      closedPidSet.add(pid);
    }
    await waitForPortToClose(remoteDebuggingPort, killTimeoutMs);
  } else if (detectedInstance.staleLock) {
    if (!options.force) {
      throw new Error(`Profile is locked by stale files: ${detectedInstance.lockFiles.join(', ')}`);
    }

    removeLockFiles(detectedInstance.lockFiles);
  }

  const launched = await launchBrowser(spec, executablePath, userDataDir, remoteDebuggingPort, startupUrl);
  debugLog('initBrowser', 'launched browser', { remoteDebuggingPort, pid: launched.pid ?? null, startupUrl });

  const wsUrl = await waitForCdpReady(remoteDebuggingPort, connectTimeoutMs);
  debugLog('initBrowser', 'waited for cdp ready', { remoteDebuggingPort, connectTimeoutMs, wsUrl });
  if (!wsUrl) {
    throw new Error(`CDP did not become available on port ${remoteDebuggingPort}`);
  }

  return {
    ok: true,
    type: spec.type,
    executablePath,
    userDataDir,
    remoteDebuggingPort,
    wsUrl,
    pid: launched.pid ?? await resolveBrowserPid(remoteDebuggingPort, detectedInstance.pid),
  };
}

export async function runConnectBrowserCommand(values: BrowserCliValues) {
  const resolved = await resolveConnectOptions({
    browser: values.browser,
    instanceName: values.instance,
    executablePath: values['executable-path'],
    userDataDir: values['user-data-dir'],
    force: values.force,
    remoteDebuggingPort: values.port ? Number(values.port) : undefined,
    connectTimeoutMs: values.timeout ? Number(values.timeout) : undefined,
    killTimeoutMs: values['kill-timeout'] ? Number(values['kill-timeout']) : undefined,
    startupUrl: values['startup-url'],
  });
  const result = await initBrowser(resolved.connectOptions);

  if (resolved.lastConnect) {
    updateLastConnect(resolved.lastConnect);
  }

  printJson(result);
}

async function main() {
  const { values } = parseBrowserCliArgs(process.argv.slice(2));

  if (values.help) {
    printConnectBrowserHelp();
    return;
  }

  await runConnectBrowserCommand(values);
}

runCli(import.meta.url, main);
