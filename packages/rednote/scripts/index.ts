#!/usr/bin/env -S node --experimental-strip-types

import { runCli } from './utils/browser-cli.ts';

function printRootHelp() {
  process.stdout.write(`rednote

Usage:
  npx -y @skills-store/rednote browser <command> [...args]
  npx -y @skills-store/rednote <command> [...args]

Commands:
  browser <list|create|remove|connect>
  env [--format md|json]
  status [--instance NAME]
  check-login [--instance NAME]
  login [--instance NAME]
  publish [--instance NAME]
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
  npx -y @skills-store/rednote search --instance seller-main --keyword 护肤
`);
}

export async function runRootCli(argv: string[] = process.argv.slice(2)) {
  const [command, ...restArgv] = argv;

  if (!command || command === 'help' || command === '--help' || command === '-h') {
    printRootHelp();
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
