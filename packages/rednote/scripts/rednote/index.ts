#!/usr/bin/env -S node --experimental-strip-types

import { parseArgs } from 'node:util';
import { runCli } from '../utils/browser-cli.ts';

function printRednoteHelp() {
  process.stdout.write(`rednote

Commands:
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
  get-feed-detail [--instance NAME] --url URL [--url URL] [--format md|json]
  get-profile [--instance NAME] --id USER_ID [--format md|json]

Examples:
  npx -y @skills-store/rednote browser list
  npx -y @skills-store/rednote browser create --name seller-main --browser chrome
  npx -y @skills-store/rednote env
  npx -y @skills-store/rednote status --instance seller-main
  npx -y @skills-store/rednote login --instance seller-main
  npx -y @skills-store/rednote publish --instance seller-main --type video --video ./note.mp4 --title 标题 --content 描述
  npx -y @skills-store/rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "写得真好"
  npx -y @skills-store/rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action like
  npx -y @skills-store/rednote home --instance seller-main --format md --save
  npx -y @skills-store/rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
  npx -y @skills-store/rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
  npx -y @skills-store/rednote get-profile --instance seller-main --id USER_ID
`);
}

function parseBasicArgs(argv: string[]) {
  const { values } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      keyword: { type: 'string' },
      format: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  return values as {
    instance?: string;
    keyword?: string;
    format?: string;
    help?: boolean;
  };
}

export async function runRednoteCli(argv: string[] = process.argv.slice(2)) {
  const rawArgv = argv;
  const firstArg = rawArgv[0];

  if (!firstArg || firstArg === 'help' || ((firstArg === '--help' || firstArg === '-h'))) {
    printRednoteHelp();
    return;
  }

  const command = !firstArg.startsWith('-') ? firstArg : 'status';
  const commandArgv = firstArg === command ? rawArgv.slice(1) : rawArgv;
  const basicValues = parseBasicArgs(commandArgv);

  if (command === 'env') {
    const { runEnvCommand } = await import('./env.ts');
    await runEnvCommand({ format: basicValues.format === 'json' ? 'json' : 'md', help: basicValues.help });
    return;
  }

  if (command === 'status') {
    const { runStatusCommand } = await import('./status.ts');
    await runStatusCommand({ instance: basicValues.instance, help: basicValues.help });
    return;
  }

  if (command === 'check-login') {
    const { runCheckLoginCommand } = await import('./checkLogin.ts');
    await runCheckLoginCommand({ instance: basicValues.instance, help: basicValues.help });
    return;
  }

  if (command === 'login') {
    const { runLoginCommand } = await import('./login.ts');
    await runLoginCommand({ instance: basicValues.instance, help: basicValues.help });
    return;
  }

  if (command === 'publish') {
    const { parsePublishCliArgs, runPublishCommand } = await import('./publish.ts');
    await runPublishCommand(parsePublishCliArgs(commandArgv));
    return;
  }

  if (command === 'comment') {
    const { parseCommentCliArgs, runCommentCommand } = await import('./comment.ts');
    await runCommentCommand(parseCommentCliArgs(commandArgv));
    return;
  }

  if (command === 'interact') {
    const { parseInteractCliArgs, runInteractCommand } = await import('./interact.ts');
    await runInteractCommand(parseInteractCliArgs(commandArgv));
    return;
  }

  if (command === 'home') {
    const { parseHomeCliArgs, runHomeCommand } = await import('./home.ts');
    await runHomeCommand(parseHomeCliArgs(commandArgv));
    return;
  }

  if (command === 'search') {
    const { parseSearchCliArgs, runSearchCommand } = await import('./search.ts');
    await runSearchCommand(parseSearchCliArgs(commandArgv));
    return;
  }

  if (command === 'get-feed-detail') {
    const { parseGetFeedDetailCliArgs, runGetFeedDetailCommand } = await import('./getFeedDetail.ts');
    await runGetFeedDetailCommand(parseGetFeedDetailCliArgs(commandArgv));
    return;
  }

  if (command === 'get-profile') {
    const { parseGetProfileCliArgs, runGetProfileCommand } = await import('./getProfile.ts');
    await runGetProfileCommand(parseGetProfileCliArgs(commandArgv));
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

async function main() {
  await runRednoteCli();
}

runCli(import.meta.url, main);
