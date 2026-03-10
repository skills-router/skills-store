---
name: rednote
description: Use when the user needs to publish, search, inspect, comment on, or otherwise operate Xiaohongshu (RedNote) from the terminal with the `@skills-store/rednote` CLI.
---

# Rednote Commands

Use this skill only for the CLI command surface of `@skills-store/rednote` when the user wants to operate Xiaohongshu from the terminal.

Focus on giving the exact command, the minimal required flags, and the right command order.

## Preferred command style

Prefer global-install examples first:

```bash
npm install -g @skills-store/rednote
bun add -g @skills-store/rednote
rednote <command> [...args]
```

Only mention `npx -y @skills-store/rednote ...` if the user explicitly asks for one-off execution without global installation.

Only show local repo commands if the user is explicitly developing the CLI.

## Core workflow

Use this sequence for most live Xiaohongshu operations:

1. `rednote env`
2. `rednote browser list` or `rednote browser create`
3. `rednote browser connect`
4. `rednote login` or `rednote check-login`
5. `rednote status --instance seller-main`
6. Operate with `home`, `search`, `get-feed-detail`, `get-profile`, `publish`, `comment`, or `interact`

If the user needs exact browser subcommands, flags, or examples, open `references/browser.md`.

If the instance is blocked by a stale profile lock, check `references/browser.md` for the force reconnect command.

## Common use cases

### Find posts from home feed

Read the current recommendation feed:

```bash
rednote home --instance seller-main --format md
rednote home --instance seller-main --format json --save ./output/home.jsonl
```

Use `home` when the user wants to browse candidate posts from the personalized feed before choosing one to inspect or comment on.

### Find posts by keyword

Search by keyword:

```bash
rednote search --instance seller-main --keyword 护肤
rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
```

Use `search` when the user wants candidate notes for a topic instead of the home feed.

### Get one note's detail

Fetch one note by URL:

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --format json
```

Use `get-feed-detail` after `home` or `search` when the user wants the title,正文,互动数据,图片/视频, and existing comments before taking an action.

### Comment on one note

Post a comment by note URL:

```bash
rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "内容写得很清楚，步骤也很实用，感谢分享。"
```

Use `comment` when the user wants to open the note detail page, type into the comment box, and click the send button.

### Interact with one note

Perform one note interaction by URL:

```bash
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action like
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action collect
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action comment --content "内容写得很清楚，步骤也很实用，感谢分享。"
```

Use `interact` when the user wants one command entrypoint for like, collect, or comment. Use `interact --action comment` when the user wants the same behavior as `comment` but through a unified interface.

### Publish content

Publish content for an authenticated instance.

Video note:

```bash
rednote publish --instance seller-main --type video --video ./note.mp4 --title 标题 --content 描述 --tag 穿搭 --tag 日常 --publish
```

Image note:

```bash
rednote publish --instance seller-main --type image --image ./1.jpg --image ./2.jpg --title 标题 --content 描述 --tag 探店 --publish
```

Article:

```bash
rednote publish --instance seller-main --type article --title 标题 --content $'# 一级标题\n\n正文' --publish
```

Use `publish` when the user wants to post or save drafts from an authenticated browser instance.

## End-to-end examples

### Home → detail → comment

Use this flow when the user wants to discover a post first and comment after reading the detail:

```bash
rednote home --instance seller-main --format md
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "谢谢分享，信息整理得很完整，对我很有帮助。"
```

### Home → detail → like or collect

Use this flow when the user wants to inspect a post before liking or collecting it:

```bash
rednote home --instance seller-main --format md
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action like
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action collect
```

### Search → detail → comment

Use this flow when the user starts from a topic keyword:

```bash
rednote search --instance seller-main --keyword 低糖早餐 --format md
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --format json
rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "这份搭配看起来很实用，食材和步骤都写得很清楚。"
```

### Inspect profile after finding a post

Use this flow when the user wants the author context before engaging:

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --format json
rednote get-profile --instance seller-main --id USER_ID
```

## Command reference

### `browser`

Use `browser list` to inspect default browsers, custom instances, stale locks, and the current `lastConnect` target.

Use `browser create` to create a reusable named browser profile for later account-scoped commands.

For exact `browser` subcommands, flags, and examples, open `references/browser.md`.

### `env`

```bash
rednote env
rednote env --format json
```

Use `env` when the user is debugging installation or local setup.

### `status`

```bash
rednote status --instance seller-main
```

Use `status` to confirm whether the instance exists, is running, and appears logged in.

### `check-login`

```bash
rednote check-login --instance seller-main
```

Use `check-login` when the user only wants to verify whether the session is still valid.

### `login`

```bash
rednote login --instance seller-main
```

Use `login` after `browser connect` if the account is not authenticated yet.

### `home`

```bash
rednote home --instance seller-main --format md --save
```

Use `home` when the user wants the current home feed and optionally wants to save it.

### `search`

```bash
rednote search --instance seller-main --keyword 护肤
rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
```

Use `search` when the user wants notes by keyword.

### `get-feed-detail`

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

Use `get-feed-detail` when the user already has a note URL and wants structured detail data.

### `get-profile`

```bash
rednote get-profile --instance seller-main --id USER_ID
```

Use `get-profile` when the user wants author or account profile information.

### `comment`

```bash
rednote comment --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --content "内容写得很清楚，感谢分享。"
```

Use `comment` when the user wants to leave a reply on a specific note.

### `interact`

```bash
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action like
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action collect
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --action comment --content "内容写得很清楚，感谢分享。"
```

Use `interact` when the user wants a unified command for like, collect, or comment.

## JSON success shapes

When the user asks for JSON output, use these success shapes as the stable mental model.

### Common note item shape

`home`, `search`, and `profile.notes` share the normalized `RednotePost` shape:

```json
{
  "id": "string",
  "modelType": "string",
  "xsecToken": "string|null",
  "url": "string",
  "noteCard": {
    "type": "string|null",
    "displayTitle": "string|null",
    "cover": {
      "urlDefault": "string|null",
      "urlPre": "string|null",
      "url": "string|null",
      "fileId": "string|null",
      "width": "number|null",
      "height": "number|null",
      "infoList": [{ "imageScene": "string|null", "url": "string|null" }]
    },
    "user": {
      "userId": "string|null",
      "nickname": "string|null",
      "nickName": "string|null",
      "avatar": "string|null",
      "xsecToken": "string|null"
    },
    "interactInfo": {
      "liked": "boolean",
      "likedCount": "string|null",
      "commentCount": "string|null",
      "collectedCount": "string|null",
      "sharedCount": "string|null"
    },
    "cornerTagInfo": [{ "type": "string|null", "text": "string|null" }],
    "imageList": [{ "width": "number|null", "height": "number|null", "infoList": [{ "imageScene": "string|null", "url": "string|null" }] }],
    "video": { "duration": "number|null" }
  }
}
```

### `env --format json`

`env` is the main exception: success JSON is a raw environment object, not `{ "ok": true, ... }`.

```json
{
  "packageRoot": "string",
  "homeDir": "string",
  "platform": "string",
  "nodeVersion": "string",
  "storageHome": "string",
  "storageRoot": "string",
  "instancesDir": "string",
  "instanceStorePath": "string",
  "legacyPackageInstancesDir": "string"
}
```

### Browser commands

`browser list`:

```json
{
  "lastConnect": { "scope": "default|custom", "name": "string", "browser": "chrome|edge|chromium|brave" } | null,
  "instances": [{
    "type": "chrome|edge|chromium|brave",
    "name": "string",
    "executablePath": "string",
    "userDataDir": "string",
    "exists": true,
    "inUse": false,
    "pid": "number|null",
    "lockFiles": ["string"],
    "matchedProcess": { "pid": "number", "name": "string", "cmdline": "string" } | null,
    "staleLock": false,
    "remotePort": "number|null",
    "scope": "default|custom",
    "instanceName": "string",
    "createdAt": "string|null",
    "lastConnect": false
  }]
}
```

`browser create`:

```json
{
  "ok": true,
  "instance": {
    "name": "string",
    "browser": "chrome|edge|chromium|brave",
    "userDataDir": "string",
    "createdAt": "string",
    "remoteDebuggingPort": "number|undefined"
  }
}
```

`browser connect`:

```json
{
  "ok": true,
  "type": "chrome|edge|chromium|brave",
  "executablePath": "string",
  "userDataDir": "string",
  "remoteDebuggingPort": "number",
  "wsUrl": "string",
  "pid": "number|null"
}
```

`browser remove`:

```json
{
  "ok": true,
  "removedInstance": {
    "name": "string",
    "browser": "chrome|edge|chromium|brave",
    "userDataDir": "string",
    "createdAt": "string",
    "remoteDebuggingPort": "number|undefined"
  },
  "removedDir": true,
  "closedPids": ["number"]
}
```

### Session and account commands

`status`:

```json
{
  "ok": true,
  "instance": {
    "scope": "default|custom",
    "name": "string",
    "browser": "chrome|edge|chromium|brave",
    "source": "argument|last-connect|single-instance",
    "status": "running|stopped|missing|stale-lock",
    "exists": true,
    "inUse": false,
    "pid": "number|null",
    "remotePort": "number|null",
    "userDataDir": "string",
    "createdAt": "string|null",
    "lastConnect": false
  },
  "rednote": {
    "loginStatus": "logged-in|logged-out|unknown",
    "lastLoginAt": "string|null"
  }
}
```

`check-login`:

```json
{
  "ok": true,
  "rednote": {
    "loginStatus": "logged-in|logged-out|unknown",
    "lastLoginAt": "string|null",
    "needLogin": false,
    "checkedAt": "string"
  }
}
```

`login`:

```json
{
  "ok": true,
  "rednote": {
    "loginClicked": true,
    "pageUrl": "string",
    "waitingForPhoneLogin": true,
    "message": "string"
  }
}
```

### Feed and profile commands

`home --format json`:

```json
{
  "ok": true,
  "home": {
    "pageUrl": "string",
    "fetchedAt": "string",
    "total": "number",
    "posts": ["RednotePost"],
    "savedPath": "string|undefined"
  }
}
```

`search --format json`:

```json
{
  "ok": true,
  "search": {
    "keyword": "string",
    "pageUrl": "string",
    "fetchedAt": "string",
    "total": "number",
    "posts": ["RednotePost"],
    "savedPath": "string|undefined"
  }
}
```

`get-feed-detail --format json`:

```json
{
  "ok": true,
  "detail": {
    "fetchedAt": "string",
    "total": "number",
    "items": [{
      "url": "string",
      "note": {
        "noteId": "string|null",
        "title": "string|null",
        "desc": "string|null",
        "type": "string|null",
        "interactInfo": {
          "liked": "boolean|null",
          "likedCount": "string|null",
          "commentCount": "string|null",
          "collected": "boolean|null",
          "collectedCount": "string|null",
          "shareCount": "string|null",
          "followed": "boolean|null"
        },
        "tagList": [{ "name": "string|null" }],
        "imageList": [{ "urlDefault": "string|null", "urlPre": "string|null", "width": "number|null", "height": "number|null" }],
        "video": { "url": "string|null", "raw": "unknown" } | null,
        "raw": "unknown"
      },
      "comments": [{
        "id": "string|null",
        "content": "string|null",
        "userId": "string|null",
        "nickname": "string|null",
        "likedCount": "string|null",
        "subCommentCount": "number|null",
        "raw": "unknown"
      }]
    }]
  }
}
```

`get-profile --format json`:

```json
{
  "ok": true,
  "profile": {
    "userId": "string",
    "url": "string",
    "fetchedAt": "string",
    "user": {
      "userId": "string|null",
      "nickname": "string|null",
      "desc": "string|null",
      "avatar": "string|null",
      "ipLocation": "string|null",
      "gender": "string|null",
      "follows": "string|number|null",
      "fans": "string|number|null",
      "interaction": "string|number|null",
      "tags": ["string"],
      "raw": "unknown"
    },
    "notes": ["RednotePost"],
    "raw": {
      "userPageData": "unknown",
      "notes": "unknown"
    }
  }
}
```

### Action commands

`publish`:

```json
{
  "ok": true,
  "message": "string"
}
```

`comment`:

```json
{
  "ok": true,
  "comment": {
    "url": "string",
    "content": "string",
    "commentedAt": "string"
  }
}
```

`interact`:

```json
{
  "ok": true,
  "message": "string"
}
```

## Flag guidance

- `--instance NAME` picks the browser instance for account-scoped commands.
- `--format json` is best for scripting.
- `--format md` is best for direct reading.
- `--save` is useful for `home` and `search` when the user wants saved output.
- `--keyword` is required for `search`.
- `--url` is required for `get-feed-detail`, `comment`, and `interact`.
- `--content` is required for `comment`, and also for `interact` when `--action comment`.
- `--action` is required for `interact`.
- `--id` is required for `get-profile`.
- `--type`, `--title`, `--content`, `--video`, `--image`, `--tag`, and `--publish` are the main `publish` flags.
- `publish` usually requires a connected and logged-in instance before running; without `--publish`, it saves a draft.

## Response style

When answering users:

- lead with the exact command they should run
- include only the flags needed for that task
- prefer one happy-path example first
- mention `browser connect` and `login` if the command requires an authenticated instance
- recommend `home` or `search` first when the user still needs to find a post
- recommend `get-feed-detail` before `comment` when the user wants to inspect the post before replying

## Troubleshooting

If a command fails, check these in order:

- the instance name is correct
- the browser instance was created or connected
- login was completed for that instance
- the profile was not blocked by a stale lock; if it was, run `rednote browser connect --instance NAME --force`
- the user provided the required flag such as `--keyword`, `--url`, `--content`, or `--id`
