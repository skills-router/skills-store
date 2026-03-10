#!/usr/bin/env -S node --experimental-strip-types

import { createRequire } from 'node:module';

import { runCli } from './utils/browser-cli.ts';

const require = createRequire(import.meta.url);
const { version } = require('../package.json') as { version: string };

function printRootHelp() {
  process.stdout.write(`rednote
Version:
  ${version}

Usage:
  npx -y @skills-store/rednote browser <command> [...args]
  npx -y @skills-store/rednote <command> [...args]

Commands:
  --version
  browser <list|create|remove|connect>
  env [--format md|json]
  status [--instance NAME]
  check-login [--instance NAME]
  login [--instance NAME]
  publish [--instance NAME]
  comment [--instance NAME] --url URL --content TEXT
  interact [--instance NAME] --url URL --action like|collect|comment [--content TEXT]
  home [--instance NAME] [--format md|json] [--save [PATH]]
  search [--instance NAME] --keyword TEXT [--format md|json] [--save [PATH]]
  get-feed-detail [--instance NAME] --url URL [--format md|json]
  get-profile [--instance NAME] --id USER_ID [--format md|json]

Examples:
  npx -y @skills-store/rednote browser list
  npx -y @skills-store/rednote browser create --name seller-main --browser chrome
  npx -y @skills-store/rednote browser connect --instance seller-main
  npx -y @skills-store/rednote env
  npx -y @skills-store/rednote publish --instance seller-main --type video --video ./note.mp4 --title 标题 --content 描述
  npx -y @skills-store/rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "写得真好"
  npx -y @skills-store/rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action like
  npx -y @skills-store/rednote search --instance seller-main --keyword 护肤
`);
}

export async function runRootCli(argv: string[] = process.argv.slice(2)) {
  const [command, ...restArgv] = argv;

  if (!command || command === 'help' || command === '--help' || command === '-h') {
    printRootHelp();
    return;
  }

  if (command === '--version' || command === '-v' || command === 'version') {
    process.stdout.write(`${version}\n`);
    return;
  }

  if (command === 'browser') {
    const { runBrowserCli } = await import('./browser/index.ts');
    await runBrowserCli(restArgv);
    return;
  }

  const { runRednoteCli } = await import('./rednote/index.ts');
  await runRednoteCli(argv);
}

async function main() {
  await runRootCli();
}

runCli(import.meta.url, main);
