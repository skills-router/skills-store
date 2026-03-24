import path from 'node:path';
import { createClient } from '@libsql/client';
import { nanoid } from 'nanoid';
import { DataSource, EntitySchema, In } from 'typeorm';
import { REDNOTE_DATABASE_PATH, ensureDir } from '../utils/browser-core.ts';
import type { RednoteDetailNote } from './getFeedDetail.ts';
import type { RednoteProfileResult } from './getProfile.ts';
import type { RednotePost } from './post-types.ts';

type JsonValue = Record<string, unknown> | unknown[] | string | number | boolean | null;

type RednotePostRecord = {
  id: string;
  noteId: string;
  title: string | null;
  url: string | null;
  image: string | null;
  likeCount: string | null;
  commentCount: string | null;
  collectedCount: string | null;
  sharedCount: string | null;
  authorId: string | null;
  authorNickname: string | null;
  modelType: string | null;
  xsecToken: string | null;
  instanceName: string;
  raw: JsonValue;
  createdAt: Date;
  updatedAt: Date;
};

type RednotePostDetailRecord = {
  id: string;
  noteId: string;
  title: string | null;
  content: string | null;
  tags: string[];
  imageList: string[];
  videoUrl: string | null;
  likeCount: string | null;
  commentCount: string | null;
  collectedCount: string | null;
  shareCount: string | null;
  authorId: string | null;
  authorNickname: string | null;
  noteType: string | null;
  instanceName: string;
  raw: JsonValue;
  createdAt: Date;
  updatedAt: Date;
};

type RednotePostCommentRecord = {
  id: string;
  commentId: string;
  noteId: string;
  content: string | null;
  likeCount: string | null;
  replyCount: string | null;
  authorId: string | null;
  authorNickname: string | null;
  instanceName: string;
  raw: JsonValue;
  createdAt: Date;
  updatedAt: Date;
};

const rednotePostSchema = new EntitySchema<RednotePostRecord>({
  name: 'RednotePostRecord',
  tableName: 'rednote_posts',
  columns: {
    id: { type: String, primary: true, length: 16 },
    noteId: { name: 'noteid', type: String },
    title: { type: String, nullable: true },
    url: { type: String, nullable: true },
    image: { type: String, nullable: true },
    likeCount: { name: 'like_count', type: String, nullable: true },
    commentCount: { name: 'comment_count', type: String, nullable: true },
    collectedCount: { name: 'collected_count', type: String, nullable: true },
    sharedCount: { name: 'shared_count', type: String, nullable: true },
    authorId: { name: 'author_id', type: String, nullable: true },
    authorNickname: { name: 'author_nickname', type: String, nullable: true },
    modelType: { name: 'model_type', type: String, nullable: true },
    xsecToken: { name: 'xsec_token', type: String, nullable: true },
    instanceName: { name: 'instance_name', type: String },
    raw: { type: 'simple-json', nullable: true },
    createdAt: { name: 'created_at', type: Date, createDate: true },
    updatedAt: { name: 'updated_at', type: Date, updateDate: true },
  },
  indices: [
    { name: 'IDX_rednote_posts_note_instance', columns: ['noteId', 'instanceName'], unique: true },
    { name: 'IDX_rednote_posts_instance', columns: ['instanceName'] },
  ],
});

const rednotePostDetailSchema = new EntitySchema<RednotePostDetailRecord>({
  name: 'RednotePostDetailRecord',
  tableName: 'rednote_post_details',
  columns: {
    id: { type: String, primary: true, length: 16 },
    noteId: { name: 'noteid', type: String },
    title: { type: String, nullable: true },
    content: { type: 'text', nullable: true },
    tags: { type: 'simple-json', nullable: true },
    imageList: { name: 'image_list', type: 'simple-json', nullable: true },
    videoUrl: { name: 'video_url', type: String, nullable: true },
    likeCount: { name: 'like_count', type: String, nullable: true },
    commentCount: { name: 'comment_count', type: String, nullable: true },
    collectedCount: { name: 'collected_count', type: String, nullable: true },
    shareCount: { name: 'share_count', type: String, nullable: true },
    authorId: { name: 'author_id', type: String, nullable: true },
    authorNickname: { name: 'author_nickname', type: String, nullable: true },
    noteType: { name: 'note_type', type: String, nullable: true },
    instanceName: { name: 'instance_name', type: String },
    raw: { type: 'simple-json', nullable: true },
    createdAt: { name: 'created_at', type: Date, createDate: true },
    updatedAt: { name: 'updated_at', type: Date, updateDate: true },
  },
  indices: [
    { name: 'IDX_rednote_post_details_note_instance', columns: ['noteId', 'instanceName'], unique: true },
    { name: 'IDX_rednote_post_details_instance', columns: ['instanceName'] },
  ],
});

const rednotePostCommentSchema = new EntitySchema<RednotePostCommentRecord>({
  name: 'RednotePostCommentRecord',
  tableName: 'rednote_post_comments',
  columns: {
    id: { type: String, primary: true, length: 16 },
    commentId: { name: 'commentid', type: String },
    noteId: { name: 'noteid', type: String },
    content: { type: 'text', nullable: true },
    likeCount: { name: 'like_count', type: String, nullable: true },
    replyCount: { name: 'reply_count', type: String, nullable: true },
    authorId: { name: 'author_id', type: String, nullable: true },
    authorNickname: { name: 'author_nickname', type: String, nullable: true },
    instanceName: { name: 'instance_name', type: String },
    raw: { type: 'simple-json', nullable: true },
    createdAt: { name: 'created_at', type: Date, createDate: true },
    updatedAt: { name: 'updated_at', type: Date, updateDate: true },
  },
  indices: [
    { name: 'IDX_rednote_post_comments_unique', columns: ['commentId', 'noteId', 'instanceName'], unique: true },
    { name: 'IDX_rednote_post_comments_note_instance', columns: ['noteId', 'instanceName'] },
  ],
});

type RednoteProfileRecord = {
  id: string;
  userId: string;
  nickname: string | null;
  desc: string | null;
  avatar: string | null;
  ipLocation: string | null;
  gender: string | null;
  follows: string | null;
  fans: string | null;
  interaction: string | null;
  tags: string[];
  fetchedAt: Date;
  instanceName: string;
  raw: JsonValue;
  createdAt: Date;
};

const rednoteProfileSchema = new EntitySchema<RednoteProfileRecord>({
  name: 'RednoteProfileRecord',
  tableName: 'rednote_profiles',
  columns: {
    id: { type: String, primary: true, length: 16 },
    userId: { name: 'userid', type: String },
    nickname: { type: String, nullable: true },
    desc: { type: 'text', nullable: true },
    avatar: { type: String, nullable: true },
    ipLocation: { name: 'ip_location', type: String, nullable: true },
    gender: { type: String, nullable: true },
    follows: { type: String, nullable: true },
    fans: { type: String, nullable: true },
    interaction: { type: String, nullable: true },
    tags: { type: 'simple-json', nullable: true },
    fetchedAt: { name: 'fetched_at', type: Date },
    instanceName: { name: 'instance_name', type: String },
    raw: { type: 'simple-json', nullable: true },
    createdAt: { name: 'created_at', type: Date, createDate: true },
  },
  indices: [
    { name: 'IDX_rednote_profiles_userid_instance', columns: ['userId', 'instanceName'] },
    { name: 'IDX_rednote_profiles_fetched_at', columns: ['fetchedAt'] },
    { name: 'IDX_rednote_profiles_instance', columns: ['instanceName'] },
  ],
});

let dataSourcePromise: Promise<DataSource> | null = null;

function createRecordId() {
  return nanoid(16);
}

function uniqueStrings(values: Array<string | null | undefined>) {
  return [...new Set(values.filter((value): value is string => typeof value === 'string' && value.length > 0))];
}

function firstNonEmpty(...values: Array<string | null | undefined>) {
  return values.find((value) => typeof value === 'string' && value.length > 0) ?? null;
}

function toCountString(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === '') {
    return null;
  }

  return String(value);
}

function coalesceValue<T>(nextValue: T | null | undefined, fallbackValue: T | null | undefined) {
  return nextValue ?? fallbackValue ?? null;
}

function extractPrimaryImage(post: RednotePost) {
  const cover = post.noteCard.cover;
  return firstNonEmpty(
    cover.urlDefault,
    cover.urlPre,
    cover.url,
    ...post.noteCard.imageList.flatMap((image) => image.infoList.map((info) => info.url)),
  );
}

function extractNoteIdFromUrl(url: string) {
  try {
    const parsed = new URL(url);
    const segments = parsed.pathname.split('/').filter(Boolean);
    return segments.at(-1) ?? null;
  } catch {
    return null;
  }
}

function extractAuthorFromRawNote(rawNote: JsonValue) {
  if (!rawNote || typeof rawNote !== 'object' || Array.isArray(rawNote)) {
    return {
      userId: null,
      nickname: null,
    };
  }

  const note = rawNote as {
    user?: { userId?: string | null; user_id?: string | null; nickname?: string | null; nickName?: string | null; nick_name?: string | null };
    userInfo?: { userId?: string | null; user_id?: string | null; nickname?: string | null; nickName?: string | null; nick_name?: string | null };
    user_info?: { userId?: string | null; user_id?: string | null; nickname?: string | null; nickName?: string | null; nick_name?: string | null };
  };
  const user = note.user ?? note.userInfo ?? note.user_info;

  return {
    userId: user?.userId ?? user?.user_id ?? null,
    nickname: user?.nickname ?? user?.nickName ?? user?.nick_name ?? null,
  };
}

function extractCommentId(comment: any) {
  const value = comment?.id ?? comment?.commentId ?? comment?.comment_id;
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value);
  }

  const fallbackUserId = comment?.userInfo?.userId ?? comment?.user_info?.user_id ?? 'unknown';
  const fallbackCreateTime = comment?.createTime ?? comment?.create_time ?? 'unknown';
  const fallbackContent = comment?.content ?? '';
  return `${fallbackUserId}:${fallbackCreateTime}:${fallbackContent}`;
}

async function initializeDataSource() {
  ensureDir(path.dirname(REDNOTE_DATABASE_PATH));
  const client = createClient({
    url: `file:${REDNOTE_DATABASE_PATH}`,
  });
  await client.execute('SELECT 1');
  client.close();

  const dataSource = new DataSource({
    type: 'better-sqlite3',
    database: REDNOTE_DATABASE_PATH,
    entities: [rednotePostSchema, rednotePostDetailSchema, rednotePostCommentSchema, rednoteProfileSchema],
    synchronize: true,
    logging: false,
    prepareDatabase: (database) => {
      database.pragma('journal_mode = WAL');
      database.pragma('foreign_keys = ON');
    },
  });

  return await dataSource.initialize();
}

export async function initializeRednoteDatabase() {
  if (!dataSourcePromise) {
    dataSourcePromise = initializeDataSource().catch((error) => {
      dataSourcePromise = null;
      throw error;
    });
  }

  return await dataSourcePromise;
}

type PersistPostInput = {
  post: RednotePost;
  raw: JsonValue;
};

async function persistPosts(instanceName: string, inputs: PersistPostInput[]) {
  if (inputs.length === 0) {
    return;
  }

  const dataSource = await initializeRednoteDatabase();
  const repository = dataSource.getRepository(rednotePostSchema);
  const noteIds = uniqueStrings(inputs.map((input) => input.post.id));
  const existingRows = noteIds.length > 0
    ? await repository.find({
        where: {
          instanceName,
          noteId: In(noteIds),
        },
      })
    : [];
  const existingMap = new Map(existingRows.map((row) => [row.noteId, row]));

  const entities = inputs.map(({ post, raw }) => {
    const existing = existingMap.get(post.id);
    const image = extractPrimaryImage(post);
    const authorNickname = firstNonEmpty(post.noteCard.user.nickname, post.noteCard.user.nickName);
    return repository.create({
      id: existing?.id ?? createRecordId(),
      noteId: post.id,
      title: coalesceValue(post.noteCard.displayTitle, existing?.title),
      url: coalesceValue(post.url, existing?.url),
      image: coalesceValue(image, existing?.image),
      likeCount: coalesceValue(toCountString(post.noteCard.interactInfo.likedCount), existing?.likeCount),
      commentCount: coalesceValue(toCountString(post.noteCard.interactInfo.commentCount), existing?.commentCount),
      collectedCount: coalesceValue(toCountString(post.noteCard.interactInfo.collectedCount), existing?.collectedCount),
      sharedCount: coalesceValue(toCountString(post.noteCard.interactInfo.sharedCount), existing?.sharedCount),
      authorId: coalesceValue(post.noteCard.user.userId, existing?.authorId),
      authorNickname: coalesceValue(authorNickname, existing?.authorNickname),
      modelType: coalesceValue(post.modelType, existing?.modelType),
      xsecToken: coalesceValue(post.xsecToken, existing?.xsecToken),
      instanceName,
      raw,
      ...(existing?.createdAt ? { createdAt: existing.createdAt } : {}),
    });
  });

  await repository.save(entities);
}

export async function persistHomePosts(instanceName: string, inputs: PersistPostInput[]) {
  await persistPosts(instanceName, inputs);
}

export async function persistSearchPosts(instanceName: string, inputs: PersistPostInput[]) {
  await persistPosts(instanceName, inputs);
}

type PersistFeedDetailInput = {
  instanceName: string;
  url: string;
  note: RednoteDetailNote;
  rawNote: JsonValue;
  rawComments?: any[];
};

export async function persistFeedDetail(input: PersistFeedDetailInput) {
  const noteId = input.note.noteId ?? extractNoteIdFromUrl(input.url);
  if (!noteId) {
    return;
  }

  const dataSource = await initializeRednoteDatabase();

  await dataSource.transaction(async (manager) => {
    const postRepository = manager.getRepository(rednotePostSchema);
    const detailRepository = manager.getRepository(rednotePostDetailSchema);
    const commentRepository = manager.getRepository(rednotePostCommentSchema);

    const [existingPost, existingDetail] = await Promise.all([
      postRepository.findOne({ where: { instanceName: input.instanceName, noteId } }),
      detailRepository.findOne({ where: { instanceName: input.instanceName, noteId } }),
    ]);
    const author = extractAuthorFromRawNote(input.rawNote);

    await postRepository.save(postRepository.create({
      id: existingPost?.id ?? createRecordId(),
      noteId,
      title: coalesceValue(input.note.title, existingPost?.title),
      url: coalesceValue(input.url, existingPost?.url),
      image: coalesceValue(input.note.imageList[0] ?? null, existingPost?.image),
      likeCount: coalesceValue(toCountString(input.note.likedCount), existingPost?.likeCount),
      commentCount: coalesceValue(toCountString(input.note.commentCount), existingPost?.commentCount),
      collectedCount: coalesceValue(toCountString(input.note.collectedCount), existingPost?.collectedCount),
      sharedCount: coalesceValue(toCountString(input.note.shareCount), existingPost?.sharedCount),
      authorId: coalesceValue(author.userId, existingPost?.authorId),
      authorNickname: coalesceValue(author.nickname, existingPost?.authorNickname),
      modelType: coalesceValue(input.note.type, existingPost?.modelType),
      xsecToken: coalesceValue((() => {
        try {
          return new URL(input.url).searchParams.get('xsec_token');
        } catch {
          return null;
        }
      })(), existingPost?.xsecToken),
      instanceName: input.instanceName,
      raw: input.rawNote,
      ...(existingPost?.createdAt ? { createdAt: existingPost.createdAt } : {}),
    }));

    await detailRepository.save(detailRepository.create({
      id: existingDetail?.id ?? createRecordId(),
      noteId,
      title: coalesceValue(input.note.title, existingDetail?.title),
      content: coalesceValue(input.note.desc, existingDetail?.content),
      tags: input.note.tagList.length > 0 ? input.note.tagList : (existingDetail?.tags ?? []),
      imageList: input.note.imageList.length > 0 ? input.note.imageList : (existingDetail?.imageList ?? []),
      videoUrl: coalesceValue(input.note.video, existingDetail?.videoUrl),
      likeCount: coalesceValue(toCountString(input.note.likedCount), existingDetail?.likeCount),
      commentCount: coalesceValue(toCountString(input.note.commentCount), existingDetail?.commentCount),
      collectedCount: coalesceValue(toCountString(input.note.collectedCount), existingDetail?.collectedCount),
      shareCount: coalesceValue(toCountString(input.note.shareCount), existingDetail?.shareCount),
      authorId: coalesceValue(author.userId, existingDetail?.authorId),
      authorNickname: coalesceValue(author.nickname, existingDetail?.authorNickname),
      noteType: coalesceValue(input.note.type, existingDetail?.noteType),
      instanceName: input.instanceName,
      raw: input.rawNote,
      ...(existingDetail?.createdAt ? { createdAt: existingDetail.createdAt } : {}),
    }));

    const rawComments = Array.isArray(input.rawComments) ? input.rawComments : [];
    if (rawComments.length === 0) {
      return;
    }

    const commentIds = rawComments.map(extractCommentId);
    const existingComments = await commentRepository.find({
      where: {
        instanceName: input.instanceName,
        noteId,
        commentId: In(commentIds),
      },
    });
    const existingCommentMap = new Map(existingComments.map((comment) => [comment.commentId, comment]));

    const commentEntities = rawComments.map((comment) => {
      const commentId = extractCommentId(comment);
      const existing = existingCommentMap.get(commentId);
      return commentRepository.create({
        id: existing?.id ?? createRecordId(),
        commentId,
        noteId,
        content: coalesceValue(comment?.content ?? null, existing?.content),
        likeCount: coalesceValue(toCountString(comment?.likeCount ?? comment?.like_count ?? comment?.interactInfo?.likedCount), existing?.likeCount),
        replyCount: coalesceValue(toCountString(comment?.subCommentCount ?? comment?.sub_comment_count), existing?.replyCount),
        authorId: coalesceValue(comment?.userInfo?.userId ?? comment?.user_info?.user_id ?? null, existing?.authorId),
        authorNickname: coalesceValue(comment?.userInfo?.nickname ?? comment?.user_info?.nickname ?? null, existing?.authorNickname),
        instanceName: input.instanceName,
        raw: comment,
        ...(existing?.createdAt ? { createdAt: existing.createdAt } : {}),
      });
    });

    await commentRepository.save(commentEntities);
  });
}


export type PersistedPostSummary = {
  id: string;
  noteId: string;
  title: string | null;
  likeCount: string | null;
  url: string | null;
};

export async function listPersistedPostSummaries(instanceName: string, noteIds: string[]): Promise<PersistedPostSummary[]> {
  const uniqueNoteIds = uniqueStrings(noteIds);
  if (uniqueNoteIds.length === 0) {
    return [];
  }

  const dataSource = await initializeRednoteDatabase();
  const repository = dataSource.getRepository(rednotePostSchema);
  const rows = await repository.find({
    where: {
      instanceName,
      noteId: In(uniqueNoteIds),
    },
  });
  const rowMap = new Map(rows.map((row) => [row.noteId, row]));

  return uniqueNoteIds
    .map((noteId) => rowMap.get(noteId))
    .filter((row): row is RednotePostRecord => Boolean(row))
    .map((row) => ({
      id: row.id,
      noteId: row.noteId,
      title: row.title,
      likeCount: row.likeCount,
      url: row.url,
    }));
}

export async function findPersistedPostUrlByRecordId(instanceName: string, id: string): Promise<string | null> {
  const dataSource = await initializeRednoteDatabase();
  const repository = dataSource.getRepository(rednotePostSchema);
  const row = await repository.findOne({
    where: {
      instanceName,
      id,
    },
  });

  return row?.url ?? null;
}

export type PersistProfileInput = {
  instanceName: string;
  result: RednoteProfileResult;
};

export async function persistProfile(input: PersistProfileInput): Promise<void> {
  const { instanceName, result } = input;
  const { profile } = result;
  const { user, notes, userId, url, fetchedAt } = profile;

  const dataSource = await initializeRednoteDatabase();

  await dataSource.transaction(async (manager) => {
    const profileRepository = manager.getRepository(rednoteProfileSchema);
    const postRepository = manager.getRepository(rednotePostSchema);

    // Insert new profile record (always insert, never update - for history tracking)
    await profileRepository.save(profileRepository.create({
      id: createRecordId(),
      userId,
      nickname: user.nickname,
      desc: user.desc,
      avatar: user.avatar,
      ipLocation: user.ipLocation,
      gender: user.gender,
      follows: toCountString(user.follows),
      fans: toCountString(user.fans),
      interaction: toCountString(user.interaction),
      tags: user.tags,
      fetchedAt: new Date(fetchedAt),
      instanceName,
      raw: user,
    }));

    // Persist notes (upsert)
    if (notes.length > 0) {
      const noteIds = uniqueStrings(notes.map((note) => note.id));
      const existingRows = noteIds.length > 0
        ? await postRepository.find({
            where: {
              instanceName,
              noteId: In(noteIds),
            },
          })
        : [];
      const existingMap = new Map(existingRows.map((row) => [row.noteId, row]));

      const entities = notes.map((post) => {
        const existing = existingMap.get(post.id);
        const image = extractPrimaryImage(post);
        const authorNickname = firstNonEmpty(post.noteCard.user.nickname, post.noteCard.user.nickName);
        return postRepository.create({
          id: existing?.id ?? createRecordId(),
          noteId: post.id,
          title: coalesceValue(post.noteCard.displayTitle, existing?.title),
          url: coalesceValue(post.url, existing?.url),
          image: coalesceValue(image, existing?.image),
          likeCount: coalesceValue(toCountString(post.noteCard.interactInfo.likedCount), existing?.likeCount),
          commentCount: coalesceValue(toCountString(post.noteCard.interactInfo.commentCount), existing?.commentCount),
          collectedCount: coalesceValue(toCountString(post.noteCard.interactInfo.collectedCount), existing?.collectedCount),
          sharedCount: coalesceValue(toCountString(post.noteCard.interactInfo.sharedCount), existing?.sharedCount),
          authorId: coalesceValue(post.noteCard.user.userId, existing?.authorId),
          authorNickname: coalesceValue(authorNickname, existing?.authorNickname),
          modelType: coalesceValue(post.modelType, existing?.modelType),
          xsecToken: coalesceValue(post.xsecToken, existing?.xsecToken),
          instanceName,
          raw: post,
          ...(existing?.createdAt ? { createdAt: existing.createdAt } : {}),
        });
      });

      await postRepository.save(entities);
    }
  });
}

export type ProfileHistoryRecord = {
  id: string;
  userId: string;
  nickname: string | null;
  desc: string | null;
  avatar: string | null;
  ipLocation: string | null;
  gender: string | null;
  follows: string | null;
  fans: string | null;
  interaction: string | null;
  tags: string[];
  fetchedAt: Date;
  instanceName: string;
  createdAt: Date;
};

export async function getProfileHistory(
  instanceName: string,
  userId: string,
  options?: { limit?: number },
): Promise<ProfileHistoryRecord[]> {
  const dataSource = await initializeRednoteDatabase();
  const repository = dataSource.getRepository(rednoteProfileSchema);
  const rows = await repository.find({
    where: { instanceName, userId },
    order: { fetchedAt: 'DESC' },
    take: options?.limit ?? 100,
  });

  return rows.map((row) => ({
    id: row.id,
    userId: row.userId,
    nickname: row.nickname,
    desc: row.desc,
    avatar: row.avatar,
    ipLocation: row.ipLocation,
    gender: row.gender,
    follows: row.follows,
    fans: row.fans,
    interaction: row.interaction,
    tags: row.tags,
    fetchedAt: row.fetchedAt,
    instanceName: row.instanceName,
    createdAt: row.createdAt,
  }));
}
