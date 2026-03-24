#!/usr/bin/env -S node --experimental-strip-types

import * as cheerio from 'cheerio';
import { parseArgs } from 'node:util';
import vm from 'node:vm';
import type { Page, Response } from 'playwright-core';
import { runCli } from '../utils/browser-cli.ts';
import { simulateMouseMove, simulateMouseWheel } from '../utils/mouse-helper.ts';
import { resolveStatusTarget } from './status.ts';
import { createRednoteSession, disconnectRednoteSession, ensureRednoteLoggedIn, type RednoteSession } from './checkLogin.ts';
import type { RednotePost } from './post-types.ts';
import { ensureJsonSavePath, renderJsonSaveSummary, renderPostsMarkdown, resolveJsonSavePath, writeJsonFile } from './output-format.ts';
import { persistProfile } from './persistence.ts';

export type ProfileFormat = 'json' | 'md';

export type ProfileMode = 'profile' | 'notes';

export type GetProfileCliValues = {
  instance?: string;
  id?: string;
  format: ProfileFormat;
  mode: ProfileMode;
  maxNotes: number;
  savePath?: string;
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
  // raw: unknown;
};

export type RednoteProfileResult = {
  ok: true;
  profile: {
    userId: string;
    url: string;
    fetchedAt: string;
    user: RednoteProfileUser;
    notes: RednotePost[];
    // raw: {
    //   userPageData: unknown;
    //   notes: unknown;
    // };
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
  npx -y @skills-store/rednote get-profile [--instance NAME] --id USER_ID [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]
  node --experimental-strip-types ./scripts/rednote/getProfile.ts --instance NAME --id USER_ID [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]
  bun ./scripts/rednote/getProfile.ts --instance NAME --id USER_ID [--mode profile|notes] [--max-notes N] [--format md|json] [--save PATH]

Options:
  --instance NAME   Optional. Defaults to the saved lastConnect instance
  --id USER_ID      Required. Xiaohongshu profile user id
  --mode MODE       Optional. profile | notes. Default: profile
  --max-notes N     Optional. Max notes to fetch by scrolling. Default: 100
  --format FORMAT   Output format: md | json. Default: md
  --save PATH       Required when --format json is used. Saves only the selected mode data as JSON
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
      mode: { type: 'string' },
      'max-notes': { type: 'string' },
      save: { type: 'string' },
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

  const mode = values.mode ?? 'profile';
  if (mode !== 'profile' && mode !== 'notes') {
    throw new Error(`Invalid --mode value: ${String(values.mode)}`);
  }

  const maxNotesValue = values['max-notes'] ?? '100';
  const maxNotes = parseInt(maxNotesValue, 10);
  if (isNaN(maxNotes) || maxNotes < 1) {
    throw new Error(`Invalid --max-notes value: ${maxNotesValue}`);
  }

  return {
    instance: values.instance,
    id: values.id,
    format,
    mode,
    maxNotes,
    savePath: values.save,
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
  };
}

import { buildExploreUrl, decodeUrlEscapedValue } from './url-format.ts';

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
  const xsecToken = decodeUrlEscapedValue(firstNonNull(item.xsecToken, item.xsec_token));

  return {
    id,
    modelType: firstNonNull(item.modelType, item.model_type) ?? 'note',
    xsecToken,
    url: buildExploreUrl(id, xsecToken),
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

function renderProfileUserMarkdown(result: RednoteProfileResult) {
  const { user, url, userId } = result.profile;
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

  return `${lines.join('\n')}\n`;
}

export function selectProfileOutput(result: RednoteProfileResult, mode: ProfileMode) {
  return mode === 'notes' ? result.profile.notes : result.profile.user;
}

export function renderProfileMarkdown(result: RednoteProfileResult, mode: ProfileMode) {
  if (mode === 'notes') {
    return renderPostsMarkdown(result.profile.notes);
  }

  return renderProfileUserMarkdown(result);
}

const NOTES_CONTAINER_SELECTOR = '.feeds-tab-container';
const NOTES_SCROLL_TIMEOUT_MS = 60_000;
const NOTES_SCROLL_IDLE_LIMIT = 4;

async function scrollNotesContainer(page: Page, maxNotes: number, getNoteCount: () => number) {
  const container = page.locator(NOTES_CONTAINER_SELECTOR).first();
  const visible = await container.isVisible().catch(() => false);
  if (!visible) {
    return;
  }

  await container.scrollIntoViewIfNeeded().catch(() => {});
  await simulateMouseMove(page, { locator: container, settleMs: 100 }).catch(() => {});

  const getMetrics = async () => await container.evaluate((element) => {
    const htmlElement = element as HTMLElement;
    const atBottom = htmlElement.scrollTop + htmlElement.clientHeight >= htmlElement.scrollHeight - 8;
    return {
      scrollTop: htmlElement.scrollTop,
      scrollHeight: htmlElement.scrollHeight,
      clientHeight: htmlElement.clientHeight,
      atBottom,
    };
  }).catch(() => null);

  const deadline = Date.now() + NOTES_SCROLL_TIMEOUT_MS;
  let idleRounds = 0;

  while (Date.now() < deadline) {
    if (getNoteCount() >= maxNotes) {
      return;
    }

    const beforeMetrics = await getMetrics();
    if (!beforeMetrics) {
      return;
    }

    const beforeCount = getNoteCount();
    const delta = Math.max(Math.floor(beforeMetrics.clientHeight * 0.85), 480);
    await simulateMouseWheel(page, { locator: container, deltaY: delta, moveBeforeScroll: false, settleMs: 900 }).catch(() => {});

    const afterMetrics = await getMetrics();
    await page.waitForTimeout(400);
    const afterCount = getNoteCount();

    const countChanged = afterCount > beforeCount;
    const scrollMoved = Boolean(afterMetrics) && afterMetrics.scrollTop > beforeMetrics.scrollTop;
    const reachedBottom = Boolean(afterMetrics?.atBottom);

    if (countChanged || scrollMoved) {
      idleRounds = 0;
      continue;
    }

    idleRounds += 1;
    if ((reachedBottom && idleRounds >= 2) || idleRounds >= NOTES_SCROLL_IDLE_LIMIT) {
      return;
    }
  }
}

async function captureProfile(page: Page, targetUrl: string, maxNotes: number) {
  let userPageData: any = null;
  const notesMap = new Map<string, any>();

  const handleResponse = async (response: Response) => {
    try {
      const url = new URL(response.url());
      if (response.status() !== 200 || !(url.href.includes('/user/profile/') || url.href.includes('/api/sns/web/v1/user_posted'))) {
        return;
      }
      if(url.href.includes('/user/profile/')){
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

        const notesData = sandbox.info?.user?.notes;
        if (Array.isArray(notesData)) {
          for (const note of notesData) {
            if(!Array.isArray(notesData)){
              const noteId = note?.id ?? note?.noteId ?? note?.note_id;
              if (noteId && !notesMap.has(noteId)) {
                notesMap.set(noteId, note);
              }
            }else{
              for (const _note of note) {
                const noteId = _note?.id ?? _note?.noteId ?? _note?.note_id;
              if (noteId && !notesMap.has(noteId)) {
                notesMap.set(noteId, _note);
              }
              }

            }
            
          }
        }
      });
      }

      if(url.href.includes('/api/sns/web/v1/user_posted')){
        const body = await response.json();
        if(body.code == 0 && body.data?.notes){
          if (Array.isArray(body.data?.notes)) {
            for (const note of body.data?.notes) {
              const noteId = note?.id ?? note?.noteId ?? note?.note_id;
              if (noteId && !notesMap.has(noteId)) {
                notesMap.set(noteId, note);
              }
            }
          }
        }
      }
    } catch {
    }
  };

  page.on('response', handleResponse);
  try {
    if(targetUrl !== page.url()){
      await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });
    }else{
      // await page.reload({ waitUntil: 'domcontentloaded' });
    }
    

    const deadline = Date.now() + 15_000;
    while (Date.now() < deadline) {
      if (userPageData || notesMap.size > 0) {
        break;
      }
      await page.waitForTimeout(200);
    }

    if (!userPageData && notesMap.size === 0) {
      throw new Error(`Failed to capture profile detail: ${targetUrl}`);
    }

    // Scroll to load more notes if needed
    if (notesMap.size < maxNotes) {
      await scrollNotesContainer(page, maxNotes, () => notesMap.size);
    }

    return {
      userPageData,
      notes: [...notesMap.values()],
    };
  } finally {
    page.off('response', handleResponse);
  }
}

export async function getProfile(session: RednoteSession, url: string, userId: string, maxNotes = 100): Promise<RednoteProfileResult> {
  validateProfileUrl(url);
  const page = await getOrCreateXiaohongshuPage(session);
  const captured = await captureProfile(page, url, maxNotes);

  return {
    ok: true,
    profile: {
      userId,
      url,
      fetchedAt: new Date().toISOString(),
      user: normalizeProfileUser({...captured.userPageData, userId }),
      notes: normalizeProfileNotes(captured.notes),
      // raw: captured,
    },
  };
}

function writeProfileOutput(result: RednoteProfileResult, values: GetProfileCliValues) {
  const output = selectProfileOutput(result, values.mode);

  if (values.format === 'json') {
    const savedPath = resolveJsonSavePath(values.savePath);
    writeJsonFile(output, savedPath);
    process.stdout.write(renderJsonSaveSummary(savedPath, output));
    return;
  }

  process.stdout.write(renderProfileMarkdown(result, values.mode));
}

export async function runGetProfileCommand(values: GetProfileCliValues = { format: 'md', mode: 'profile', maxNotes: 100 }) {
  if (values.help) {
    printGetProfileHelp();
    return;
  }

  ensureJsonSavePath(values.format, values.savePath);

  if (!values.id) {
    throw new Error('Missing required option: --id');
  }

  const target = resolveStatusTarget(values.instance);
  const normalizedUserId = values.id.trim();
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'fetching profile', session);
    const result = await getProfile(session, buildProfileUrl(normalizedUserId), normalizedUserId, values.maxNotes);

    // Persist profile and notes to database
    await persistProfile({
      instanceName: target.instanceName,
      result,
    });

    writeProfileOutput(result, values);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parseGetProfileCliArgs(process.argv.slice(2));
  await runGetProfileCommand(values);
}

runCli(import.meta.url, main);
