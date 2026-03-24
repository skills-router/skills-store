import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import { execFile, spawn, spawnSync } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import chromePath from 'chrome-paths';
import { getEdgePath } from 'edge-paths';
import psList from 'ps-list';
import { portToPid } from 'pid-port';
import { stringifyError } from './browser-cli.ts';

const execFileAsync = promisify(execFile);
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export type BrowserType = 'chrome' | 'edge' | 'chromium' | 'brave';

export type BrowserSpec = {
  type: BrowserType;
  displayName: string;
  executableCandidates: string[];
  userDataDir: string;
  processNames: string[];
  pathCommands: string[];
};

export type ProcessInfo = {
  pid: number;
  name: string;
  cmdline: string;
};

export type BrowserInstanceInfo = {
  type: BrowserType;
  name: string;
  executablePath: string;
  userDataDir: string;
  exists: boolean;
  inUse: boolean;
  pid: number | null;
  lockFiles: string[];
  matchedProcess: ProcessInfo | null;
  staleLock: boolean;
  remotePort: number | null;
};

export type InspectedBrowserInstance = BrowserInstanceInfo & {
  pids: number[];
  matchedProcesses: ProcessInfo[];
};

export type PersistedInstance = {
  name: string;
  browser: BrowserType;
  userDataDir: string;
  createdAt: string;
  remoteDebuggingPort?: number;
};

export type LastConnectInfo = {
  scope: 'default' | 'custom';
  name: string;
  browser: BrowserType;
};

export type InstanceStore = {
  version: 1;
  lastConnect: LastConnectInfo | null;
  instances: PersistedInstance[];
};

export type ListedBrowserInstance = BrowserInstanceInfo & {
  scope: 'default' | 'custom';
  instanceName: string;
  createdAt: string | null;
  lastConnect: boolean;
};

export type ConnectBrowserOptions = {
  browser?: BrowserType;
  executablePath?: string;
  userDataDir?: string;
  force?: boolean;
  remoteDebuggingPort?: number;
  connectTimeoutMs?: number;
  killTimeoutMs?: number;
  startupUrl?: string;
};

export type ConnectBrowserResult = {
  ok: true;
  type: BrowserType;
  instanceName?: string;
  executablePath: string;
  userDataDir: string;
  remoteDebuggingPort: number;
  wsUrl: string;
  pid: number | null;
};

export type CreateInstanceOptions = {
  name: string;
  browser?: BrowserType;
  remoteDebuggingPort?: number;
};

export type CreateInstanceResult = {
  ok: true;
  instance: PersistedInstance;
};

export type RemoveInstanceOptions = {
  name: string;
  force?: boolean;
};

export type RemoveInstanceResult = {
  ok: true;
  removedInstance: PersistedInstance;
  removedDir: boolean;
  closedPids: number[];
};

export type ListBrowserInstancesResult = {
  lastConnect: LastConnectInfo | null;
  instances: ListedBrowserInstance[];
};

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
export const PACKAGE_ROOT = path.resolve(SCRIPT_DIR, '../..');
export const SKILLS_ROUTER_HOME = path.join(os.homedir(), '.skills-router');
export const REDNOTE_STORAGE_ROOT = path.join(SKILLS_ROUTER_HOME, 'rednote');
export const REDNOTE_DATABASE_PATH = path.join(REDNOTE_STORAGE_ROOT, 'main.db');
export const INSTANCES_DIR = path.join(REDNOTE_STORAGE_ROOT, 'instances');
export const INSTANCE_STORE_PATH = path.join(INSTANCES_DIR, 'data.json');
export const LEGACY_PACKAGE_INSTANCES_DIR = path.join(PACKAGE_ROOT, 'instances');

export function normalizePath(inputPath: string) {
  let resolved = path.resolve(inputPath);
  if (process.platform === 'win32') {
    resolved = resolved.toLowerCase();
  }
  return resolved.replace(/[\\/]+$/, '');
}

function unique<T>(values: T[]) {
  return [...new Set(values)];
}

function defaultInstanceStore(): InstanceStore {
  return {
    version: 1,
    lastConnect: null,
    instances: [],
  };
}

function currentManagedInstanceDir(name: string) {
  return path.join(INSTANCES_DIR, name);
}

function legacyManagedInstanceDirs(name: string) {
  return [
    path.join(LEGACY_PACKAGE_INSTANCES_DIR, name),
    path.resolve(PACKAGE_ROOT, '../../skills/rednote/instances', name),
  ];
}

function resolvePersistedInstanceUserDataDir(name: string, userDataDir: string) {
  const normalizedInput = normalizePath(userDataDir);
  const managedCurrentDir = currentManagedInstanceDir(name);

  if (normalizedInput === normalizePath(managedCurrentDir)) {
    return managedCurrentDir;
  }

  if (legacyManagedInstanceDirs(name).some((dirPath) => normalizedInput === normalizePath(dirPath))) {
    return managedCurrentDir;
  }

  const instancesSuffix = path.sep + 'instances' + path.sep + name;
  if (normalizedInput.endsWith(instancesSuffix)) {
    return managedCurrentDir;
  }

  return userDataDir;
}

export function exists(filePath: string) {
  try {
    fs.accessSync(filePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function migrateLegacyInstanceStoreIfNeeded() {
  if (normalizePath(LEGACY_PACKAGE_INSTANCES_DIR) === normalizePath(INSTANCES_DIR)) {
    return;
  }

  if (exists(INSTANCE_STORE_PATH) || !exists(LEGACY_PACKAGE_INSTANCES_DIR)) {
    return;
  }

  fs.mkdirSync(REDNOTE_STORAGE_ROOT, { recursive: true });
  fs.cpSync(LEGACY_PACKAGE_INSTANCES_DIR, INSTANCES_DIR, {
    recursive: true,
    force: false,
    errorOnExist: false,
  });
}

function ensureInstanceStoreDir() {
  migrateLegacyInstanceStoreIfNeeded();
  fs.mkdirSync(INSTANCES_DIR, { recursive: true });
}

export function getRednoteEnvironmentInfo() {
  return {
    packageRoot: PACKAGE_ROOT,
    homeDir: os.homedir(),
    platform: process.platform,
    nodeVersion: process.version,
    storageHome: SKILLS_ROUTER_HOME,
    storageRoot: REDNOTE_STORAGE_ROOT,
    databasePath: REDNOTE_DATABASE_PATH,
    instancesDir: INSTANCES_DIR,
    instanceStorePath: INSTANCE_STORE_PATH,
    legacyPackageInstancesDir: LEGACY_PACKAGE_INSTANCES_DIR,
  };
}

export function readInstanceStore(): InstanceStore {
  ensureInstanceStoreDir();

  if (!exists(INSTANCE_STORE_PATH)) {
    return defaultInstanceStore();
  }

  try {
    const raw = fs.readFileSync(INSTANCE_STORE_PATH, 'utf8');
    const parsed = JSON.parse(raw) as Partial<InstanceStore>;

    return {
      version: 1,
      lastConnect:
        parsed.lastConnect &&
        (parsed.lastConnect.scope === 'default' || parsed.lastConnect.scope === 'custom') &&
        typeof parsed.lastConnect.name === 'string' &&
        typeof parsed.lastConnect.browser === 'string'
          ? {
              scope: parsed.lastConnect.scope,
              name: parsed.lastConnect.name,
              browser: parsed.lastConnect.browser as BrowserType,
            }
          : null,
      instances: Array.isArray(parsed.instances)
        ? parsed.instances.flatMap((item) => {
            if (
              item &&
              typeof item.name === 'string' &&
              typeof item.browser === 'string' &&
              typeof item.userDataDir === 'string' &&
              typeof item.createdAt === 'string'
            ) {
              return [{
                name: item.name,
                browser: item.browser as BrowserType,
                userDataDir: resolvePersistedInstanceUserDataDir(item.name, item.userDataDir),
                createdAt: item.createdAt,
                remoteDebuggingPort:
                  typeof item.remoteDebuggingPort === 'number' && Number.isInteger(item.remoteDebuggingPort) && item.remoteDebuggingPort > 0
                    ? item.remoteDebuggingPort
                    : undefined,
              } satisfies PersistedInstance];
            }
            return [];
          })
        : [],
    };
  } catch {
    return defaultInstanceStore();
  }
}

export function writeInstanceStore(store: InstanceStore) {
  ensureInstanceStoreDir();
  fs.writeFileSync(INSTANCE_STORE_PATH, `${JSON.stringify(store, null, 2)}
`, 'utf8');
}

export function updateInstanceRemoteDebuggingPort(name: string, remoteDebuggingPort: number) {
  const store = readInstanceStore();
  const instanceName = validateInstanceName(name);
  const nextInstances = store.instances.map((instance) =>
    instance.name === instanceName
      ? {
          ...instance,
          remoteDebuggingPort,
        }
      : instance,
  );

  if (!nextInstances.some((instance) => instance.name === instanceName)) {
    throw new Error(`Instance not found: ${instanceName}`);
  }

  writeInstanceStore({
    ...store,
    instances: nextInstances,
  });
}

export async function getRandomAvailablePort() {
  return await new Promise<number>((resolve, reject) => {
    const server = net.createServer();

    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        server.close(() => reject(new Error('Failed to allocate a random port')));
        return;
      }

      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(address.port);
      });
    });
  });
}

export function customInstanceUserDataDir(name: string) {
  return path.join(INSTANCES_DIR, name);
}

export function isSubPath(parentPath: string, childPath: string) {
  const normalizedParent = normalizePath(parentPath);
  const normalizedChild = normalizePath(childPath);

  return normalizedChild === normalizedParent || normalizedChild.startsWith(`${normalizedParent}${path.sep}`);
}

export function validateInstanceName(name: string) {
  const trimmed = name.trim();
  if (!trimmed) {
    throw new Error('Instance name cannot be empty');
  }

  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{1,63}$/.test(trimmed)) {
    throw new Error('Instance name must match /^[a-zA-Z0-9][a-zA-Z0-9._-]{1,63}$/');
  }

  return trimmed;
}

export function isLastConnectMatch(
  lastConnect: LastConnectInfo | null,
  scope: 'default' | 'custom',
  instanceName: string,
  browser: BrowserType,
) {
  return Boolean(
    lastConnect &&
      lastConnect.scope === scope &&
      lastConnect.name === instanceName &&
      lastConnect.browser === browser,
  );
}

export function updateLastConnect(lastConnect: LastConnectInfo) {
  const store = readInstanceStore();
  writeInstanceStore({
    ...store,
    lastConnect,
  });
}

function basenameMatches(procName: string, names: string[]) {
  const base = path.basename(procName).toLowerCase();
  return names.some((name) => base === name.toLowerCase());
}

function spawnSyncText(command: string, args: string[]) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    windowsHide: true,
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(result.stderr || `Command failed: ${command} ${args.join(' ')}`);
  }

  return result.stdout;
}

function commandExists(command: string) {
  try {
    const probe = process.platform === 'win32' ? 'where' : 'command';
    const args = process.platform === 'win32' ? [command] : ['-v', command];
    const result = spawnSync(probe, args, { stdio: 'ignore', shell: process.platform !== 'win32' });
    return result.status === 0;
  } catch {
    return false;
  }
}

function findExecutableInPath(commands: string[]) {
  for (const command of commands) {
    if (!commandExists(command)) {
      continue;
    }

    try {
      if (process.platform === 'win32') {
        const output = spawnSyncText('where', [command]).trim().split(/\r?\n/)[0];
        if (output) {
          return output;
        }
      } else {
        const output = spawnSyncText('command', ['-v', command]);
        const resolved = output.trim().split(/\r?\n/)[0];
        if (resolved) {
          return resolved;
        }
      }
    } catch {
      continue;
    }
  }

  return null;
}

export function browserSpecs(): BrowserSpec[] {
  const homeDir = os.homedir();

  return [
    {
      type: 'chrome',
      displayName: 'Google Chrome',
      executableCandidates: process.platform === 'darwin'
        ? ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
        : process.platform === 'win32'
          ? [
              'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
              'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
            ]
          : ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable'],
      userDataDir: process.platform === 'darwin'
        ? path.join(homeDir, 'Library/Application Support/Google/Chrome')
        : process.platform === 'win32'
          ? path.join(process.env.LOCALAPPDATA || path.join(homeDir, 'AppData/Local'), 'Google/Chrome/User Data')
          : path.join(homeDir, '.config/google-chrome'),
      processNames: ['Google Chrome', 'Google Chrome Helper', 'chrome.exe', 'google-chrome', 'google-chrome-stable'],
      pathCommands: ['google-chrome', 'google-chrome-stable'],
    },
    {
      type: 'edge',
      displayName: 'Microsoft Edge',
      executableCandidates: process.platform === 'darwin'
        ? ['/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge']
        : process.platform === 'win32'
          ? [
              'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
              'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
            ]
          : ['/usr/bin/microsoft-edge', '/usr/bin/microsoft-edge-stable'],
      userDataDir: process.platform === 'darwin'
        ? path.join(homeDir, 'Library/Application Support/Microsoft Edge')
        : process.platform === 'win32'
          ? path.join(process.env.LOCALAPPDATA || path.join(homeDir, 'AppData/Local'), 'Microsoft/Edge/User Data')
          : path.join(homeDir, '.config/microsoft-edge'),
      processNames: ['Microsoft Edge', 'msedge.exe', 'microsoft-edge', 'microsoft-edge-stable'],
      pathCommands: ['microsoft-edge', 'microsoft-edge-stable'],
    },
    {
      type: 'chromium',
      displayName: 'Chromium',
      executableCandidates: process.platform === 'darwin'
        ? ['/Applications/Chromium.app/Contents/MacOS/Chromium']
        : process.platform === 'win32'
          ? [
              'C:\\Program Files\\Chromium\\Application\\chrome.exe',
              'C:\\Program Files (x86)\\Chromium\\Application\\chrome.exe',
            ]
          : ['/usr/bin/chromium', '/usr/bin/chromium-browser'],
      userDataDir: process.platform === 'darwin'
        ? path.join(homeDir, 'Library/Application Support/Chromium')
        : process.platform === 'win32'
          ? path.join(process.env.LOCALAPPDATA || path.join(homeDir, 'AppData/Local'), 'Chromium/User Data')
          : path.join(homeDir, '.config/chromium'),
      processNames: ['Chromium', 'chromium', 'chromium-browser', 'chrome.exe'],
      pathCommands: ['chromium', 'chromium-browser'],
    },
    {
      type: 'brave',
      displayName: 'Brave',
      executableCandidates: process.platform === 'darwin'
        ? ['/Applications/Brave Browser.app/Contents/MacOS/Brave Browser']
        : process.platform === 'win32'
          ? [
              'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
              'C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
            ]
          : ['/usr/bin/brave-browser', '/usr/bin/brave'],
      userDataDir: process.platform === 'darwin'
        ? path.join(homeDir, 'Library/Application Support/BraveSoftware/Brave-Browser')
        : process.platform === 'win32'
          ? path.join(process.env.LOCALAPPDATA || path.join(homeDir, 'AppData/Local'), 'BraveSoftware/Brave-Browser/User Data')
          : path.join(homeDir, '.config/BraveSoftware/Brave-Browser'),
      processNames: ['Brave Browser', 'brave.exe', 'brave-browser', 'brave'],
      pathCommands: ['brave-browser', 'brave'],
    },
  ];
}

export function resolveExecutablePath(spec: BrowserSpec) {
  if (spec.type === 'chrome' && chromePath?.chrome) {
    return chromePath.chrome;
  }

  if (spec.type === 'edge') {
    const edge = getEdgePath();
    if (edge) {
      return edge;
    }
  }

  for (const candidate of spec.executableCandidates) {
    if (exists(candidate)) {
      return candidate;
    }
  }

  return findExecutableInPath(spec.pathCommands);
}

export async function listProcesses(): Promise<ProcessInfo[]> {
  try {
    return (await psList()).map((proc) => ({
      pid: proc.pid,
      name: proc.name,
      cmdline: proc.cmd ?? '',
    }));
  } catch {
    return [];
  }
}

function extractUserDataDirFromCmdline(cmdline: string) {
  const patterns = [
    /--user-data-dir=(?:"([^"]+)"|'([^']+)'|(.+?))(?=\s--[a-zA-Z]|$)/i,
    /--user-data-dir\s+(?:"([^"]+)"|'([^']+)'|(.+?))(?=\s--[a-zA-Z]|$)/i,
  ];

  for (const pattern of patterns) {
    const match = cmdline.match(pattern);
    if (!match) {
      continue;
    }

    const value = match[1] || match[2] || match[3] || null;
    return value ? value.trim() : null;
  }

  return null;
}

function extractRemoteDebuggingPortFromCmdline(cmdline: string) {
  const patterns = [
    /--remote-debugging-port=(\d+)/i,
    /--remote-debugging-port\s+(\d+)/i,
  ];

  for (const pattern of patterns) {
    const match = cmdline.match(pattern);
    if (match) {
      return Number(match[1]);
    }
  }

  return null;
}

async function findPidsUsingFiles(filePaths: string[]) {
  if (process.platform === 'win32') {
    return [];
  }

  if (!commandExists('lsof')) {
    return [];
  }

  try {
    const stdout = await execFileAsync('lsof', ['-F', 'p', '--', ...filePaths], {
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
    });

    return unique(
      stdout.stdout
        .split(/\r?\n/)
        .flatMap((line) => (line.startsWith('p') ? [Number(line.slice(1))] : []))
        .filter((pid) => Number.isInteger(pid) && pid > 0),
    );
  } catch {
    return [];
  }
}

export async function findListeningPortsByPid(pid: number) {
  try {
    const ports = await portToPid(pid, 'tcp');
    return unique(ports.filter((port) => Number.isInteger(port) && port > 0));
  } catch {
    return [];
  }
}

function lockFilesFor(userDataDir: string) {
  return [path.join(userDataDir, 'SingletonLock'), path.join(userDataDir, 'SingletonCookie'), path.join(userDataDir, 'SingletonSocket')]
    .filter((filePath) => exists(filePath));
}

function processMatchesBrowser(spec: BrowserSpec, executablePath: string, userDataDir: string, proc: ProcessInfo) {
  if (!basenameMatches(proc.name, spec.processNames) && !proc.cmdline.includes(executablePath)) {
    return false;
  }

  const cmdlineUserDataDir = extractUserDataDirFromCmdline(proc.cmdline);
  if (!cmdlineUserDataDir) {
    return normalizePath(userDataDir) === normalizePath(spec.userDataDir);
  }

  return normalizePath(cmdlineUserDataDir) === normalizePath(userDataDir);
}

function pickPrimaryProcess(processes: ProcessInfo[]) {
  return (
    processes.find((proc) => !proc.cmdline.includes('--type=')) ??
    processes.find((proc) => proc.cmdline.includes('--remote-debugging-port=')) ??
    processes[0] ??
    null
  );
}

export function removeLockFiles(lockFiles: string[]) {
  for (const filePath of lockFiles) {
    try {
      fs.rmSync(filePath, { force: true });
    } catch {
    }
  }
}

export function toBrowserInstanceInfo(instance: InspectedBrowserInstance): BrowserInstanceInfo {
  return {
    type: instance.type,
    name: instance.name,
    executablePath: instance.executablePath,
    userDataDir: instance.userDataDir,
    exists: instance.exists,
    inUse: instance.inUse,
    pid: instance.pid,
    lockFiles: instance.lockFiles,
    matchedProcess: instance.matchedProcess,
    staleLock: instance.staleLock,
    remotePort: instance.remotePort,
  };
}

export async function inspectBrowserInstance(spec: BrowserSpec, executablePath?: string, userDataDir?: string) {
  const resolvedExecutablePath = executablePath || resolveExecutablePath(spec) || spec.executableCandidates[0] || spec.displayName;
  const resolvedUserDataDir = userDataDir ? path.resolve(userDataDir) : spec.userDataDir;
  const lockFiles = lockFilesFor(resolvedUserDataDir);
  const processes = await listProcesses();
  const matchedProcesses = processes.filter((proc) => processMatchesBrowser(spec, resolvedExecutablePath, resolvedUserDataDir, proc));
  const primaryProcess = pickPrimaryProcess(matchedProcesses);
  const pids = unique([
    ...matchedProcesses.map((proc) => proc.pid),
    ...(lockFiles.length > 0 ? await findPidsUsingFiles(lockFiles) : []),
  ]);
  const listeningPorts = unique((await Promise.all(pids.map((pid) => findListeningPortsByPid(pid)))).flat());
  const cmdlineRemotePort = primaryProcess ? extractRemoteDebuggingPortFromCmdline(primaryProcess.cmdline) : null;
  const remotePort = cmdlineRemotePort ?? listeningPorts[0] ?? null;

  return {
    type: spec.type,
    name: spec.type,
    executablePath: resolvedExecutablePath,
    userDataDir: resolvedUserDataDir,
    exists: exists(resolvedUserDataDir),
    inUse: pids.length > 0,
    pid: primaryProcess?.pid ?? pids[0] ?? null,
    pids,
    lockFiles,
    matchedProcess: primaryProcess,
    matchedProcesses,
    staleLock: lockFiles.length > 0 && pids.length === 0,
    remotePort,
  } satisfies InspectedBrowserInstance;
}

export function isDefaultInstanceName(name: string) {
  return browserSpecs().some((spec) => spec.type === name);
}

export function findSpec(browserType: BrowserType) {
  const spec = browserSpecs().find((item) => item.type === browserType);
  if (!spec) {
    throw new Error(`Unsupported browser type: ${browserType}`);
  }
  return spec;
}

export async function getPidsListeningOnPort(port: number) {
  try {
    return unique(await portToPid(port, 'tcp'));
  } catch {
    return [];
  }
}

function isPidAlive(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function closePidGracefully(pid: number) {
  try {
    process.kill(pid, 'SIGTERM');
    return true;
  } catch {
    return false;
  }
}

async function killPidForce(pid: number) {
  try {
    process.kill(pid, 'SIGKILL');
    return true;
  } catch {
    return false;
  }
}

async function waitForPidExit(pid: number, timeoutMs = 8_000, intervalMs = 250) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isPidAlive(pid)) {
      return true;
    }
    await sleep(intervalMs);
  }
  return !isPidAlive(pid);
}

export async function waitForPortToClose(port: number, timeoutMs = 8_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if ((await getPidsListeningOnPort(port)).length === 0) {
      return true;
    }
    await sleep(250);
  }
  return (await getPidsListeningOnPort(port)).length === 0;
}

export async function getCdpWebSocketUrl(port: number) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/json/version`);
    if (!response.ok) {
      return null;
    }
    const data = await response.json() as { webSocketDebuggerUrl?: string };
    return data.webSocketDebuggerUrl || null;
  } catch {
    return null;
  }
}

export async function waitForCdpReady(port: number, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const wsUrl = await getCdpWebSocketUrl(port);
    if (wsUrl) {
      return wsUrl;
    }
    await sleep(250);
  }
  return null;
}

async function waitForPortBound(port: number, timeoutMs = 5_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise<void>((resolve) => {
      const socket = net.connect({ host: '127.0.0.1', port }, () => {
        socket.end();
        resolve();
      });
      socket.on('error', () => resolve());
    });

    if ((await getPidsListeningOnPort(port)).length > 0) {
      return true;
    }

    await sleep(200);
  }

  return false;
}

async function loadPlaywrightChromium() {
  const playwright = await import('playwright-core');
  return playwright.chromium;
}

export async function connectOverCdp(endpoint: number | string) {
  const chromium = await loadPlaywrightChromium();
  const url = typeof endpoint === 'number' ? `http://127.0.0.1:${endpoint}` : endpoint;
  const browser = await chromium.connectOverCDP(url);
  return browser;
}

export function ensureDir(dirPath: string) {
  fs.mkdirSync(dirPath, { recursive: true });
  return dirPath;
}

export function resolveUserDataDir(spec: BrowserSpec, options: ConnectBrowserOptions) {
  return ensureDir(options.userDataDir ? path.resolve(options.userDataDir) : spec.userDataDir);
}

export function isBrowserProcess(spec: BrowserSpec, proc: ProcessInfo) {
  return basenameMatches(proc.name, spec.processNames);
}

export async function closeProcessesByPid(pids: number[], timeoutMs: number) {
  const closedPids: number[] = [];

  for (const pid of unique(pids.filter((value) => Number.isInteger(value) && value > 0))) {
    if (!isPidAlive(pid)) {
      closedPids.push(pid);
      continue;
    }

    await closePidGracefully(pid);
    if (!(await waitForPidExit(pid, timeoutMs))) {
      await killPidForce(pid);
      await waitForPidExit(pid, timeoutMs);
    }

    if (!isPidAlive(pid)) {
      closedPids.push(pid);
    }
  }

  return closedPids;
}

export async function launchBrowser(spec: BrowserSpec, executablePath: string, userDataDir: string, port: number, startupUrl: string) {
  const browserProcess = spawn(
    executablePath,
    [
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${userDataDir}`,
      '--no-first-run',
      '--no-default-browser-check',
      startupUrl,
    ],
    {
      detached: process.platform !== 'win32',
      stdio: 'ignore',
      windowsHide: true,
      env: process.env,
    },
  );

  browserProcess.unref();
  await waitForPortBound(port, 5_000);
  const wsUrl = await waitForCdpReady(port);
  if (!wsUrl) {
    throw new Error(`Browser started but CDP did not become ready on port ${port}`);
  }

  return {
    pid: browserProcess.pid ?? null,
    wsUrl,
  };
}
