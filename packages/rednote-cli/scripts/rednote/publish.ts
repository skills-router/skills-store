#!/usr/bin/env -S node --experimental-strip-types

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import type { Locator, Page } from 'playwright-core';
import { printJson, runCli } from '../utils/browser-cli.ts';
import { resolveStatusTarget } from './status.ts';
import {
  createRednoteSession,
  disconnectRednoteSession,
  ensureRednoteLoggedIn,
  type RednoteSession,
} from './checkLogin.ts';

const REDNOTE_EXPLORE_URL = 'https://www.xiaohongshu.com/explore';
const CREATOR_HOME_URL = 'https://creator.xiaohongshu.com/new/home';
const CREATOR_SERVICE_SELECTOR = 'a.link[href="//creator.xiaohongshu.com/?source=official"]';
const CREATOR_CENTER_TRIGGER_SELECTOR = 'span.reds-button-new-text';
const CREATOR_CENTER_TRIGGER_TEXT = '创作中心';
const PUBLISH_NOTE_BUTTON_SELECTOR = 'span.btn-text';
const PUBLISH_NOTE_BUTTON_TEXT = '发布笔记';
const PUBLISH_TYPE_TAB_SELECTOR = '.header-tabs .creator-tab .title';
const VIDEO_UPLOAD_BUTTON_SELECTOR = 'button.upload-button';
const VIDEO_UPLOAD_BUTTON_TEXT = '上传视频';
const IMAGE_UPLOAD_BUTTON_SELECTOR = 'button.upload-button';
const IMAGE_UPLOAD_BUTTON_TEXT = '上传图片';
const IMAGE_PUBLISH_PAGE_SELECTOR = '.publish-page-content';
const IMAGE_TITLE_INPUT_SELECTOR = '.publish-page-content input.d-text[placeholder="填写标题会有更多赞哦"]';
const IMAGE_CONTENT_EDITOR_SELECTOR = '.editor-content .tiptap.ProseMirror[contenteditable="true"]';
const ARTICLE_NEW_BUTTON_SELECTOR = 'button.new-btn';
const ARTICLE_NEW_BUTTON_TEXT = '新的创作';
const ARTICLE_TITLE_INPUT_SELECTOR = '.edit-page textarea.d-text[placeholder="输入标题"]';
const ARTICLE_CONTENT_EDITOR_SELECTOR = '.rich-editor-content .tiptap.ProseMirror[contenteditable="true"]';
const PUBLISH_ACTION_BUTTON_SELECTOR = '.publish-page-publish-btn button';
const SAVE_DRAFT_BUTTON_TEXT = '暂存离开';
const PUBLISH_BUTTON_TEXT = '发布';
const VIDEO_FILE_INPUT_SELECTOR = 'input[type="file"]';
const IMAGE_FILE_INPUT_SELECTOR = 'input[type="file"]';
const MAX_IMAGE_COUNT = 15;

export type PublishType = 'video' | 'image' | 'article';

export type PublishCliValues = {
  instance?: string;
  type?: PublishType;
  title?: string;
  content?: string;
  tags: string[];
  videoPath?: string;
  imagePaths: string[];
  publishNow: boolean;
  help?: boolean;
};

export type ResolvedPublishPayload =
  | {
      type: 'video';
      title: string;
      content: string;
      tags: string[];
      draft: boolean;
      videoPath: string;
    }
  | {
      type: 'image';
      title: string;
      content: string;
      tags: string[];
      draft: boolean;
      imagePaths: string[];
      coverImagePath: string;
    }
  | {
      type: 'article';
      title: string;
      draft: boolean;
      content: string;
    };

export type PublishResult = {
  ok: true;
  message: string;
};

function printPublishHelp() {
  process.stdout.write(`rednote publish

Usage:
  npx -y @skills-store/rednote publish --type video --video ./video.mp4 --title "Video title" --content "Video description" [--tag fashion] [--tag ootd] [--publish] [--instance NAME]
  npx -y @skills-store/rednote publish --type image --image ./1.jpg --image ./2.jpg --title "Image title" --content "Image description" [--tag travel] [--publish] [--instance NAME]
  npx -y @skills-store/rednote publish --type article --title "Article title" --content '# Heading\n\nBody content' [--publish] [--instance NAME]
  node --experimental-strip-types ./scripts/rednote/publish.ts ...
  bun ./scripts/rednote/publish.ts ...

Options:
  --instance NAME      Optional. Defaults to the saved lastConnect instance
  --type TYPE          Optional. video | image | article. If omitted, the type is inferred from the provided assets
  --title TEXT         Required. Post title
  --content TEXT       Required. Description for video/image posts, or Markdown content for article posts
  --tag TEXT           Optional. Repeat to provide multiple tags, for example: --tag fashion --tag OOTD
  --video PATH         Required for video posts. Only one video file is accepted
  --image PATH         Required for image posts. Repeat to provide multiple images, up to ${MAX_IMAGE_COUNT}; the first image becomes the cover
  --publish            Publish immediately. If omitted, the content is saved as a draft
  -h, --help           Show this help
`);
}

function describePublishType(type: PublishType) {
  if (type === 'video') {
    return 'Upload Video';
  }

  if (type === 'image') {
    return 'Upload Image Post';
  }

  return 'Write Article';
}

function describePublishAction(draft: boolean) {
  return draft ? 'Save Draft and Leave' : 'Publish';
}

function isCreatorHomeUrl(url: string) {
  return url === CREATOR_HOME_URL || url.startsWith(`${CREATOR_HOME_URL}?`) || url.startsWith(`${CREATOR_HOME_URL}#`);
}

function ensureNonEmpty(value: string | undefined, optionName: string) {
  const normalized = value?.trim();
  if (!normalized) {
    throw new Error(`Missing required option: ${optionName}`);
  }
  return normalized;
}

function normalizeTags(tags: string[]) {
  const normalizedTags = tags
    .map((tag) => tag.trim())
    .filter(Boolean)
    .map((tag) => tag.replace(/^#+/, ''))
    .filter(Boolean);

  return [...new Set(normalizedTags)];
}

function resolveExistingFile(filePath: string, optionName: string) {
  const resolvedPath = path.resolve(filePath);
  let stat: fs.Stats;

  try {
    stat = fs.statSync(resolvedPath);
  } catch {
    throw new Error(`${optionName} file not found: ${resolvedPath}`);
  }

  if (!stat.isFile()) {
    throw new Error(`${optionName} must point to a file: ${resolvedPath}`);
  }

  return resolvedPath;
}

export function parsePublishCliArgs(argv: string[]): PublishCliValues {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: false,
    options: {
      instance: { type: 'string' },
      type: { type: 'string' },
      title: { type: 'string' },
      content: { type: 'string' },
      tag: { type: 'string', multiple: true },
      video: { type: 'string' },
      image: { type: 'string', multiple: true },
      publish: { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (positionals.length > 0) {
    throw new Error(`Unexpected positional arguments: ${positionals.join(' ')}`);
  }

  const publishType = values.type;
  if (publishType && publishType !== 'video' && publishType !== 'image' && publishType !== 'article') {
    throw new Error(`Invalid --type value: ${String(publishType)}`);
  }

  return {
    instance: values.instance,
    type: publishType,
    title: values.title,
    content: values.content,
    tags: values.tag ?? [],
    videoPath: values.video,
    imagePaths: values.image ?? [],
    publishNow: values.publish ?? false,
    help: values.help,
  };
}


function hasPublishInputs(values: PublishCliValues) {
  return Boolean(
    values.type
      || values.title?.trim()
      || values.content?.trim()
      || values.videoPath?.trim()
      || values.imagePaths.length > 0
      || values.tags.length > 0
      || values.publishNow,
  );
}

function resolvePublishType(values: PublishCliValues): PublishType {
  if (values.type) {
    return values.type;
  }

  if (values.videoPath?.trim()) {
    return 'video';
  }

  if (values.imagePaths.length > 0) {
    return 'image';
  }

  return 'article';
}

export function resolvePublishPayload(values: PublishCliValues): ResolvedPublishPayload {
  const type = resolvePublishType(values);

  const title = ensureNonEmpty(values.title, '--title');
  const tags = normalizeTags(values.tags);
  const draft = !values.publishNow;

  if (type === 'video') {
    const content = ensureNonEmpty(values.content, '--content');
    const videoPath = ensureNonEmpty(values.videoPath, '--video');

    if (values.imagePaths.length > 0) {
      throw new Error('Do not combine --type video with --image');
    }

    return {
      type,
      title,
      content,
      tags,
      draft,
      videoPath: resolveExistingFile(videoPath, '--video'),
    };
  }

  if (type === 'image') {
    const content = ensureNonEmpty(values.content, '--content');

    if (values.videoPath) {
      throw new Error('Do not combine --type image with --video');
    }
    if (values.imagePaths.length === 0) {
      throw new Error('Missing required option: --image');
    }
    if (values.imagePaths.length > MAX_IMAGE_COUNT) {
      throw new Error(`Too many images: received ${values.imagePaths.length}, maximum is ${MAX_IMAGE_COUNT}`);
    }

    const imagePaths = values.imagePaths.map((imagePath) => resolveExistingFile(imagePath, '--image'));

    return {
      type,
      title,
      content,
      tags,
      draft,
      imagePaths,
      coverImagePath: imagePaths[0],
    };
  }

  if (values.videoPath) {
    throw new Error('Do not combine --type article with --video');
  }
  if (values.imagePaths.length > 0) {
    throw new Error('Do not combine --type article with --image');
  }
  if (tags.length > 0) {
    throw new Error('Do not combine --type article with --tag');
  }

  const content = ensureNonEmpty(values.content, '--content');

  return {
    type,
    title,
    draft,
    content,
  };
}

function getSessionPages(session: RednoteSession) {
  const pages = [session.page, ...session.browserContext.pages()];
  return [...new Set(pages)];
}

async function findCreatorServicePage(session: RednoteSession) {
  for (const page of getSessionPages(session)) {
    if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
      continue;
    }

    const creatorServiceLink = page.locator(CREATOR_SERVICE_SELECTOR).filter({ hasText: '创作服务' });
    if ((await creatorServiceLink.count()) > 0) {
      return page;
    }
  }

  return null;
}

async function resolvePublishPage(session: RednoteSession) {
  const existingCreatorHomePage = getSessionPages(session).find((page) => isCreatorHomeUrl(page.url()));
  if (existingCreatorHomePage) {
    return {
      page: existingCreatorHomePage,
      reusedCreatorHome: true,
    };
  }

  const existingCreatorServicePage = await findCreatorServicePage(session);
  if (existingCreatorServicePage) {
    return {
      page: existingCreatorServicePage,
      reusedCreatorHome: false,
    };
  }

  const page = session.page;
  if (!page.url().startsWith('https://www.xiaohongshu.com/')) {
    await page.goto(REDNOTE_EXPLORE_URL, {
      waitUntil: 'domcontentloaded',
    });
  }

  await page.waitForTimeout(1_500);

  return {
    page,
    reusedCreatorHome: isCreatorHomeUrl(page.url()),
  };
}

async function waitForCreatorHome(page: Page) {
  await page.waitForURL((url) => isCreatorHomeUrl(url.toString()), {
    timeout: 15_000,
  });
  await page.waitForLoadState('domcontentloaded');
}

async function findVisibleLocator(locator: Locator, timeoutMs = 3_000) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const count = await locator.count();
    for (let index = 0; index < count; index += 1) {
      const candidate = locator.nth(index);
      if (await candidate.isVisible().catch(() => false)) {
        return candidate;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  return null;
}

async function requireVisibleLocator(locator: Locator, errorMessage: string, timeoutMs = 3_000) {
  const visibleLocator = await findVisibleLocator(locator, timeoutMs);
  if (!visibleLocator) {
    throw new Error(errorMessage);
  }

  return visibleLocator;
}

function resolvePublishTypeTabText(type: PublishType) {
  if (type === 'video') {
    return '上传视频';
  }

  if (type === 'image') {
    return '上传图文';
  }

  return '写长文';
}

async function hoverCreatorCenter(page: Page) {
  const creatorCenterTrigger = page.locator(CREATOR_CENTER_TRIGGER_SELECTOR).filter({ hasText: CREATOR_CENTER_TRIGGER_TEXT });
  const visibleCreatorCenterTrigger = await findVisibleLocator(creatorCenterTrigger);
  if (!visibleCreatorCenterTrigger) {
    return;
  }

  await visibleCreatorCenterTrigger.hover();
  await page.waitForTimeout(300);
}

async function openPublishComposer(page: Page, type: PublishType) {
  const publishNoteButton = page.locator(PUBLISH_NOTE_BUTTON_SELECTOR).filter({ hasText: PUBLISH_NOTE_BUTTON_TEXT });
  const visiblePublishNoteButton = await requireVisibleLocator(
    publishNoteButton,
    'Could not find the Publish Note button. Make sure the Creator Services home page is open.',
    15_000,
  );

  await visiblePublishNoteButton.click();

  const publishTypeTabText = resolvePublishTypeTabText(type);
  const publishTypeTab = page.locator(PUBLISH_TYPE_TAB_SELECTOR).filter({ hasText: publishTypeTabText }).last();
  const visiblePublishTypeTab = await requireVisibleLocator(
    publishTypeTab,
    `Could not find the ${describePublishType(type)} entry. Make sure the Creator Services page finished loading.`,
    15_000,
  );

  await visiblePublishTypeTab.click();
  await page.waitForTimeout(300);
}

async function uploadVideoFile(page: Page, videoPath: string) {
  const uploadVideoButton = page.locator(VIDEO_UPLOAD_BUTTON_SELECTOR).filter({ hasText: VIDEO_UPLOAD_BUTTON_TEXT });
  const visibleUploadVideoButton = await requireVisibleLocator(
    uploadVideoButton,
    'Could not find the Upload Video button. Make sure the video publishing page is open.',
    15_000,
  );

  const fileChooserPromise = page.waitForEvent('filechooser', {
    timeout: 3_000,
  }).catch(() => null);

  await visibleUploadVideoButton.click();

  const fileChooser = await fileChooserPromise;
  if (fileChooser) {
    await fileChooser.setFiles(videoPath);
    return;
  }

  const videoFileInput = page.locator(`${VIDEO_FILE_INPUT_SELECTOR}[accept*="video"], ${VIDEO_FILE_INPUT_SELECTOR}`);
  if ((await videoFileInput.count()) === 0) {
    throw new Error('Could not find the video file input. Make sure the upload component finished loading.');
  }

  await videoFileInput.first().setInputFiles(videoPath);
}

async function uploadImageFiles(page: Page, imagePaths: string[]) {
  const uploadImageButton = page.locator(IMAGE_UPLOAD_BUTTON_SELECTOR).filter({ hasText: IMAGE_UPLOAD_BUTTON_TEXT });
  const visibleUploadImageButton = await requireVisibleLocator(
    uploadImageButton,
    'Could not find the Upload Image button. Make sure the image publishing page is open.',
    15_000,
  );

  const fileChooserPromise = page.waitForEvent('filechooser', {
    timeout: 3_000,
  }).catch(() => null);

  await visibleUploadImageButton.click();

  const fileChooser = await fileChooserPromise;
  if (fileChooser) {
    await fileChooser.setFiles(imagePaths);
    return;
  }

  const imageFileInput = page.locator(`${IMAGE_FILE_INPUT_SELECTOR}[accept*="image"], ${IMAGE_FILE_INPUT_SELECTOR}`);
  if ((await imageFileInput.count()) === 0) {
    throw new Error('Could not find the image file input. Make sure the upload component finished loading.');
  }

  await imageFileInput.first().setInputFiles(imagePaths);
}

async function waitForImagePublishPage(page: Page) {
  const imagePublishPage = page.locator(IMAGE_PUBLISH_PAGE_SELECTOR);
  const visibleImagePublishPage = await requireVisibleLocator(
    imagePublishPage,
    'The image publishing page did not appear. Make sure the image upload completed successfully.',
    30_000,
  );

  await visibleImagePublishPage.waitFor({
    state: 'visible',
    timeout: 30_000,
  });
}

async function fillImageTitle(page: Page, title: string) {
  const imageTitleInput = page.locator(IMAGE_TITLE_INPUT_SELECTOR);
  const visibleImageTitleInput = await requireVisibleLocator(
    imageTitleInput,
    'Could not find the image post title input. Make sure the image publishing page finished loading.',
    15_000,
  );

  await visibleImageTitleInput.fill(title);
  await page.waitForTimeout(200);
}

async function fillImageContent(page: Page, content: string) {
  const imageContentEditor = page.locator(IMAGE_CONTENT_EDITOR_SELECTOR);
  const visibleImageContentEditor = await requireVisibleLocator(
    imageContentEditor,
    'Could not find the image post content editor. Make sure the image publishing page finished loading.',
    15_000,
  );

  await visibleImageContentEditor.fill(content);
  await page.waitForTimeout(200);
}

async function openArticleEditor(page: Page) {
  const articleNewButton = page.locator(ARTICLE_NEW_BUTTON_SELECTOR).filter({ hasText: ARTICLE_NEW_BUTTON_TEXT });
  const visibleArticleNewButton = await requireVisibleLocator(
    articleNewButton,
    'Could not find the New Creation button. Make sure the article publishing page is open.',
    15_000,
  );

  await visibleArticleNewButton.click();
  await page.waitForTimeout(300);
}

async function fillArticleTitle(page: Page, title: string) {
  const articleTitleInput = page.locator(ARTICLE_TITLE_INPUT_SELECTOR);
  const visibleArticleTitleInput = await requireVisibleLocator(
    articleTitleInput,
    'Could not find the article title input. Make sure the article editor is open.',
    15_000,
  );

  await visibleArticleTitleInput.fill(title);
  await page.waitForTimeout(200);
}

async function fillArticleContent(page: Page, content: string) {
  const articleContentEditor = page.locator(ARTICLE_CONTENT_EDITOR_SELECTOR);
  const visibleArticleContentEditor = await requireVisibleLocator(
    articleContentEditor,
    'Could not find the article content editor. Make sure the article editor is open.',
    15_000,
  );

  await visibleArticleContentEditor.fill(content);
  await page.waitForTimeout(200);
}

async function preparePublishAssets(page: Page, payload: ResolvedPublishPayload) {
  if (payload.type === 'video') {
    await uploadVideoFile(page, payload.videoPath);
    await waitForImagePublishPage(page);
    await fillImageTitle(page, payload.title);
    await fillImageContent(page, payload.content);
    return;
  }

  if (payload.type === 'image') {
    await uploadImageFiles(page, payload.imagePaths);
    await waitForImagePublishPage(page);
    await fillImageTitle(page, payload.title);
    await fillImageContent(page, payload.content);
    return;
  }

  await openArticleEditor(page);
  await fillArticleTitle(page, payload.title);
  await fillArticleContent(page, payload.content);
}

async function finalizePublish(page: Page, draft: boolean) {
  const buttonText = draft ? SAVE_DRAFT_BUTTON_TEXT : PUBLISH_BUTTON_TEXT;
  const publishActionButton = page.locator(PUBLISH_ACTION_BUTTON_SELECTOR).filter({ hasText: buttonText });
  const visiblePublishActionButton = await requireVisibleLocator(
    publishActionButton,
    `Could not find the ${describePublishAction(draft)} button. Make sure the publish page finished loading.`,
    15_000,
  );

  await visiblePublishActionButton.click();
  await page.waitForTimeout(500);
}

export async function openRednotePublish(
  session: RednoteSession,
  payload: ResolvedPublishPayload,
): Promise<PublishResult> {
  const resolved = await resolvePublishPage(session);

  let targetPage = resolved.page;
  let clickedCreatorService = false;
  let reusedCreatorHome = resolved.reusedCreatorHome || isCreatorHomeUrl(resolved.page.url());
  let openedInNewPage = false;

  if (!reusedCreatorHome) {
    await hoverCreatorCenter(resolved.page);

    const creatorServiceLink = resolved.page.locator(CREATOR_SERVICE_SELECTOR).filter({ hasText: '创作服务' });
    const visibleCreatorServiceLink = await requireVisibleLocator(
      creatorServiceLink,
      'Could not find the Creator Services entry. Open the Xiaohongshu home page and make sure the account is logged in.',
    );

    const popupPromise = session.browserContext
      .waitForEvent('page', {
        timeout: 3_000,
      })
      .catch(() => null);

    await visibleCreatorServiceLink.click();

    targetPage = (await popupPromise) ?? resolved.page;
    openedInNewPage = targetPage !== resolved.page;

    try {
      await waitForCreatorHome(targetPage);
    } catch {
      const existingCreatorHomePage = getSessionPages(session).find((page) => isCreatorHomeUrl(page.url()));
      if (!existingCreatorHomePage) {
        throw new Error(`After clicking Creator Services, the page did not navigate to ${CREATOR_HOME_URL}`);
      }

      targetPage = existingCreatorHomePage;
      openedInNewPage = targetPage !== resolved.page;
    }

    clickedCreatorService = true;
    reusedCreatorHome = false;
  }

  await openPublishComposer(targetPage, payload.type);
  await preparePublishAssets(targetPage, payload);

  await finalizePublish(targetPage, payload.draft);

  return {
    ok: true,
    message: payload.draft
      ? 'Publishing page actions completed and Save Draft and Leave was clicked. The content was saved as a draft.'
      : 'Publishing page actions completed and Publish was clicked.',
  };
}

export async function runPublishCommand(values: PublishCliValues) {
  if (values.help || !hasPublishInputs(values)) {
    printPublishHelp();
    return;
  }

  const payload = resolvePublishPayload(values);
  const target = resolveStatusTarget(values.instance);
  const session = await createRednoteSession(target);

  try {
    await ensureRednoteLoggedIn(target, 'publishing content', session);
    const result = await openRednotePublish(session, payload);
    printJson(result);
  } finally {
    await disconnectRednoteSession(session);
  }
}

async function main() {
  const values = parsePublishCliArgs(process.argv.slice(2));

  if (values.help) {
    printPublishHelp();
    return;
  }

  await runPublishCommand(values);
}

runCli(import.meta.url, main);
