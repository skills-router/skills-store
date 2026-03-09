#!/usr/bin/env -S node --experimental-strip-types

import * as cheerio from 'cheerio';
import { parseArgs } from 'node:util';
import vm from 'node:vm';
import type { Page, Response } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn, type RednoteSession } from './checkLogin.ts';
import type { RednotePost } from './post-types.ts';

export type ProfileFormat = 'json' | 'md';

export type GetProfileCliValues = {
  instance?: string;
  id?: string;
  format: ProfileFormat;
  help?: boolean;
};

export type RednoteProfileUser = {
  userId: string | null;
  nickname: string | null;
  desc: string | null;
  avatar: string | null;
  ipLocation: string | null;
  gender: string | null;
  follows: string | number | null;
  fans: string | number | null;
  interaction: string | number | null;
  tags: string[];
  raw: unknown;
};

export type RednoteProfileResult = {
  ok: true;
  profile: {
    userId: string;
    url: string;
    fetchedAt: string;
    user: RednoteProfileUser;
    notes: RednotePost[];
    raw: {
      userPageData: unknown;
      notes: unknown;
    };
  };
};

type XHSProfileNoteItem = {
  id?: string;
  noteId?: string;
  modelType?: string;
  model_type?: string;
  xsecToken?: string;
  xsec_token?: string;
  noteCard?: any;
  note_card?: any;
};

function printGetProfileHelp() {
  process.stdout.write(`rednote get-profile

Usage:
  npx -y @skills-store/rednote get-profile [--instance NAME] --id USER_ID [--format md|json]
  node --experimental-strip-types ./scripts/rednote/getProfile.ts --instance NAME --id USER_ID [--format md|json]
  bun ./scripts/rednote/getProfile.ts --instance NAME --id USER_ID [--format md|json]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --id USER_ID      Required. Xiaohongshu profile user id
  --format FORMAT   Output format: md | json. Default: md
  -h, --help        Show this help
`);
}

export function parseGetProfileCliArgs(argv: string[]): GetProfileCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      id: { type: 'string' },
      format: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  const format = values.format ?? 'md';
  if (format !== 'md' && format !== 'json') {
    throw new Error(`Invalid --format value: ${String(format)}`);
  }

  return {
    instance: values.instance,
    id: values.id,
    format,
    help: values.help,
  };
}

function buildProfileUrl(userId: string) {
  const normalizedId = userId.trim();
  if (!normalizedId) {
    throw new Error('Profile id cannot be empty');
  }

  return `https://www.xiaohongshu.com/user/profile/${normalizedId}`;
}

function validateProfileUrl(url: string) {
  try {
    const parsed = new URL(url);
    if (!parsed.href.startsWith('https://www.xiaohongshu.com/user/profile/')) {
      throw new Error(`url is not valid: ${url},must start with "https://www.xiaohongshu.com/user/profile/"`);
    }
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`url is not valid: ${url}`);
    }
    throw error;
  }
}

async function getOrCreateXiaohongshuPage(session: RednoteSession) {
  return session.page;
}

function firstNonNull<T>(...values: Array<T | null | undefined>) {
  for (const value of values) {
    if (value !== null && value !== undefined) {
      return value;
    }
  }
  return null;
}

function normalizeProfileUser(userPageData: any): RednoteProfileUser {
  const basicInfo = userPageData?.basicInfo ?? userPageData?.user ?? userPageData ?? {};
  const interactions = userPageData?.interactions ?? userPageData?.interactionInfo ?? userPageData?.fansInfo ?? {};
  const tags = Array.isArray(userPageData?.tags)
    ? userPageData.tags.filter((tag: any) =>tag.tagType == 'college').map((tag: any) => String(tag?.name ?? tag?.text ?? tag ?? '')).filter(Boolean)
    : [];

    const follows = interactions.find((item: any) => item?.type === 'follows')?.count;
    const fans = interactions.find((item: any) => item?.type === 'fans')?.count;
    const interaction = interactions.find((item: any) => item?.type === 'interaction')?.count;
  return {
    userId: firstNonNull(basicInfo.userId, basicInfo.user_id),
    nickname: firstNonNull(basicInfo.nickname, basicInfo.nickName),
    desc: firstNonNull(basicInfo.desc, basicInfo.description),
    avatar: firstNonNull(basicInfo.image, basicInfo.avatar, basicInfo.images),
    ipLocation: firstNonNull(basicInfo.ipLocation, basicInfo.ip_location),
    gender: firstNonNull(basicInfo.gender, basicInfo.genderType),
    follows: follows,
    fans: fans,
    interaction: interaction,
    tags,
    raw: userPageData,
  };
}

function normalizeProfileNote(item: XHSProfileNoteItem): RednotePost | null {
  const id = firstNonNull(item.id, item.noteId);
  if (!id) {
    return null;
  }

  const noteCard = item.noteCard ?? item.note_card ?? {};
  const user = noteCard.user ?? {};
  const interactInfo = noteCard.interactInfo ?? noteCard.interact_info ?? {};
  const cover = noteCard.cover ?? {};
  const imageList = Array.isArray(noteCard.imageList ?? noteCard.image_list)
    ? (noteCard.imageList ?? noteCard.image_list)
    : [];
  const cornerTagInfo = Array.isArray(noteCard.cornerTagInfo ?? noteCard.corner_tag_info)
    ? (noteCard.cornerTagInfo ?? noteCard.corner_tag_info)
    : [];
  const xsecToken = firstNonNull(item.xsecToken, item.xsec_token);

  return {
    id,
    modelType: firstNonNull(item.modelType, item.model_type) ?? 'note',
    xsecToken,
    url: xsecToken
      ? `https://www.xiaohongshu.com/explore/${id}?xsec_token=${xsecToken}`
      : `https://www.xiaohongshu.com/explore/${id}`,
    noteCard: {
      type: firstNonNull(noteCard.type, null),
      displayTitle: firstNonNull(noteCard.displayTitle, noteCard.display_title),
      cover: {
        urlDefault: firstNonNull(cover.urlDefault, cover.url_default),
        urlPre: firstNonNull(cover.urlPre, cover.url_pre),
        url: firstNonNull(cover.url, null),
        fileId: firstNonNull(cover.fileId, cover.file_id),
        width: firstNonNull(cover.width, null),
        height: firstNonNull(cover.height, null),
        infoList: Array.isArray(cover.infoList ?? cover.info_list)
          ? (cover.infoList ?? cover.info_list).map((info: any) => ({
              imageScene: firstNonNull(info?.imageScene, info?.image_scene),
              url: firstNonNull(info?.url, null),
            }))
          : [],
      },
      user: {
        userId: firstNonNull(user.userId, user.user_id),
        nickname: firstNonNull(user.nickname, user.nickName, user.nick_name),
        nickName: firstNonNull(user.nickName, user.nick_name, user.nickname),
        avatar: firstNonNull(user.avatar, null),
        xsecToken: firstNonNull(user.xsecToken, user.xsec_token),
      },
      interactInfo: {
        liked: Boolean(firstNonNull(interactInfo.liked, false)),
        likedCount: firstNonNull(interactInfo.likedCount, interactInfo.liked_count),
        commentCount: firstNonNull(interactInfo.commentCount, interactInfo.comment_count),
        collectedCount: firstNonNull(interactInfo.collectedCount, interactInfo.collected_count),
        sharedCount: firstNonNull(interactInfo.sharedCount, interactInfo.shared_count),
      },
      cornerTagInfo: cornerTagInfo.map((tag: any) => ({
        type: firstNonNull(tag?.type, null),
        text: firstNonNull(tag?.text, null),
      })),
      imageList: imageList.map((image: any) => ({
        width: firstNonNull(image?.width, null),
        height: firstNonNull(image?.height, null),
        infoList: Array.isArray(image?.infoList ?? image?.info_list)
          ? (image.infoList ?? image.info_list).map((info: any) => ({
              imageScene: firstNonNull(info?.imageScene, info?.image_scene),
              url: firstNonNull(info?.url, null),
            }))
          : [],
      })),
      video: {
        duration: firstNonNull(noteCard?.video?.capa?.duration, null),
      },
    },
  };
}

function normalizeProfileNotes(notesRaw: any): RednotePost[] {
  const candidates: XHSProfileNoteItem[] = [];

  if (Array.isArray(notesRaw)) {
    for (const entry of notesRaw) {
      if (Array.isArray(entry)) {
        candidates.push(...entry);
        continue;
      }
      if (Array.isArray(entry?.notes)) {
        candidates.push(...entry.notes);
        continue;
      }
      if (Array.isArray(entry?.items)) {
        candidates.push(...entry.items);
        continue;
      }
      if (entry && typeof entry === 'object') {
        candidates.push(entry);
      }
    }
  }

  return candidates.map(normalizeProfileNote).filter((item): item is RednotePost => Boolean(item));
}

function formatProfileField(value: string | number | null | undefined) {
  return value ?? '';
}

function renderProfileMarkdown(result: RednoteProfileResult) {
  const { user, notes, url, userId } = result.profile;
  const lines: string[] = [];

  lines.push('## UserInfo');
  lines.push('');
  lines.push(`- Url: ${url}`);
  lines.push(`- Nickname: ${formatProfileField(user.nickname)}`);
  lines.push(`- UserID: ${userId}`);
  lines.push(`- Desc: ${formatProfileField(user.desc)}`);
  lines.push(`- IpLocation: ${formatProfileField(user.ipLocation)}`);
  lines.push(`- Follows: ${formatProfileField(user.follows)}`);
  lines.push(`- Fans: ${formatProfileField(user.fans)}`);
  lines.push(`- Interactions: ${formatProfileField(user.interaction)}`);
  lines.push(`- Tags: ${user.tags.length > 0 ? user.tags.map((tag) => `#${tag}`).join(' ') : ''}`);
  lines.push('');
  lines.push('## Notes');
  lines.push('');

  if (notes.length === 0) {
    lines.push('- Notes not found or the profile is private');
  } else {
    notes.forEach((note, index) => {
      lines.push(`- Title: ${note.noteCard.displayTitle ?? ''}`);
      lines.push(`  Url: ${note.url}`);
      lines.push(`  Interaction: ${note.noteCard.interactInfo.likedCount ?? ''}`);
      if (index < notes.length - 1) {
        lines.push('');
      }
    });
  }

  return `${lines.join('\n')}\n`;
}
async function captureProfile(page: Page, targetUrl: string) {
  let userPageData: any = null;
  let notes: any = null;

  const handleResponse = async (response: Response) => {
    try {
      const url = new URL(response.url());
      if (response.status() !== 200 || !url.href.includes('/user/profile/')) {
        return;
      }

      const html = await response.text();
      const $ = cheerio.load(html);

      $('script').each((_, element) => {
        const scriptContent = $(element).html();
        if (!scriptContent?.includes('window.__INITIAL_STATE__')) {
          return;
        }

        const scriptText = scriptContent.substring(scriptContent.indexOf('=') + 1);
        const sandbox: { info?: any } = {};
        vm.createContext(sandbox);
        vm.runInContext(`var info = ${scriptText}`, sandbox);
        userPageData = sandbox.info?.user?.userPageData ?? userPageData;

        notes = sandbox.info?.user?.notes ?? notes;
      });
    } catch {
    }
  };

  page.on('response', handleResponse);
  try {
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });

    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      if (userPageData || notes) {
        break;
      }
      await page.waitForTimeout(200);
    }

    if (!userPageData && !notes) {
      throw new Error(`Failed to capture profile detail: ${targetUrl}`);
    }

    return {
      userPageData,
      notes,
    };
  } finally {
    page.off('response', handleResponse);
  }
}

export async function getProfile(session: RednoteSession, url: string, userId: string): Promise<RednoteProfileResult> {
  validateProfileUrl(url);
  const page = await getOrCreateXiaohongshuPage(session);
  const captured = await captureProfile(page, url);

  return {
    ok: true,
    profile: {
      userId,
      url,
      fetchedAt: new Date().toISOString(),
      user: normalizeProfileUser(captured.userPageData),
      notes: normalizeProfileNotes(captured.notes),
      raw: captured,
    },
  };
}

function writeProfileOutput(result: RednoteProfileResult, format: ProfileFormat) {
  if (format === 'json') {
    printJson(result);
    return;
  }

  process.stdout.write(renderProfileMarkdown(result));
}

export async function runGetProfileCommand(values: GetProfileCliValues = { format: 'md' }) {
  if (values.help) {
    printGetProfileHelp();
    return;
  }
  if (!values.id) {
    throw new Error('Missing required option: --id');
  }

  const target = resolveStatusTarget(values.instance);
  const normalizedUserId = values.id.trim();
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'fetching profile', session);
    const result = await getProfile(session, buildProfileUrl(normalizedUserId), normalizedUserId);
    writeProfileOutput(result, values.format);
  } finally {
    disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseGetProfileCliArgs(process.argv.slice(2));
  await runGetProfileCommand(values);
}

runCli(import.meta.url, main);
