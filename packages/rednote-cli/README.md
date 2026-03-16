# @skills-store/rednote

A Xiaohongshu (RED) automation CLI for browser session management, login, search, feed detail lookup, profile lookup, and note interactions such as like, collect, and commenting through `interact`.

## Install

### Install globally

```bash
npm install -g @skills-store/rednote
bun add -g @skills-store/rednote
```

After global installation, use the `rednote` executable:

```bash
rednote <command> [...args]
```

## Recommended command order

For most tasks, run commands in this order:

```text
1. env
2. browser list or browser create
3. browser connect
4. login or check-login
5. status
6. home, search, get-feed-detail, get-profile, or interact
```

## Quick start

```bash
rednote env
rednote browser create --name seller-main --browser chrome --port 9222
rednote browser connect --instance seller-main
rednote login --instance seller-main
rednote status --instance seller-main
rednote search --instance seller-main --keyword 护肤
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --like --collect --comment "写得真好"
```

## Commands

### `browser`

```bash
rednote browser list
rednote browser create --name seller-main --browser chrome --port 9222
rednote browser connect --instance seller-main
rednote browser connect --browser edge --user-data-dir /tmp/edge-profile --port 9223
rednote browser remove --name seller-main
```

Use `browser` for browser setup, reusable profiles, and instance management.

### `env`

```bash
rednote env
rednote env --format json
```

Use `env` first when checking installation, runtime info, or storage paths.

### `status`

```bash
rednote status --instance seller-main
```

Use `status` to confirm whether an instance exists, is running, and appears logged in.

### `check-login`

```bash
rednote check-login --instance seller-main
```

Use `check-login` when you only want to verify whether the session is still valid.

### `login`

```bash
rednote login --instance seller-main
```

Use `login` after `browser connect` if the instance is not authenticated yet.

### `home`

```bash
rednote home --instance seller-main --format md --save
```

Use `home` when you want the current home feed and optionally want to save it to disk.

The terminal output always uses the compact summary format below, even when `--format json` is selected:

```text
id=<database nanoid>
title=<post title>
like=<liked count>
```

Captured home feed posts are upserted into `~/.skills-router/rednote/main.db` (the same path returned by `rednote env`). They are stored in the `rednote_posts` table, and the printed `id` is that table's `nanoid(16)` primary key.

### `search`

```bash
rednote search --instance seller-main --keyword 护肤
rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
```

Use `search` for keyword-based note lookup.

The terminal output always uses the compact summary format below, even when `--format json` is selected:

```text
id=<database nanoid>
title=<post title>
like=<liked count>
```

Captured search results are also upserted into `~/.skills-router/rednote/main.db` in the `rednote_posts` table. The printed `id` can be passed directly to `get-feed-detail --id`.

### `get-feed-detail`

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
rednote get-feed-detail --instance seller-main --id AynQ7_utnNiW1Ytk
```

Use `get-feed-detail` when you already have a Xiaohongshu note URL, or when you have a database `id` returned by `home` or `search`. With `--id`, the CLI looks up the saved URL from `~/.skills-router/rednote/main.db` and then navigates with that raw URL.

Captured note details and comments are also upserted into `~/.skills-router/rednote/main.db` in `rednote_post_details` and `rednote_post_comments`.

### `get-profile`

```bash
rednote get-profile --instance seller-main --id USER_ID
```

Use `get-profile` when you want author or account profile information.


### `interact`

```bash
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --like --collect
rednote interact --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --like --collect --comment "写得真好"
```

Use `interact` when you want the single entrypoint for note operations such as like, collect, and comment in one command. Use `--comment TEXT` for replies; there is no standalone `comment` command.

## Important flags

- `--instance NAME` selects the browser instance for account-scoped commands.
- `--format json` is best for scripting.
- `--format md` is best for direct reading.
- `--save` is useful for `home` and `search` when you want the raw post array written to disk.
- `--keyword` is required for `search`.
- `home` and `search` always print `id/title/like` summaries to stdout; `--format json` only changes the saved file payload.
- `get-feed-detail` accepts either `--url URL` or `--id ID`.
- `--id` is required for `get-profile`.
- `--url` is required for `interact`; at least one of `--like`, `--collect`, or `--comment TEXT` must be provided.
- replies are sent with `interact --comment TEXT`.

## JSON success shapes

Use these shapes as the success model when a command returns JSON.

### Shared note item

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

`env` is the main exception: it returns a raw environment object instead of `{ "ok": true, ... }`.

```json
{
  "packageRoot": "string",
  "homeDir": "string",
  "platform": "string",
  "nodeVersion": "string",
  "storageHome": "string",
  "storageRoot": "string",
  "databasePath": "string",
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
    "qrCodePath": "string|null",
    "message": "string"
  }
}
```

### Feed and profile commands

`home` stdout (both `md` and `json`):

```text
id=<database nanoid>
title=<post title>
like=<liked count>
```

`home --format json --save PATH` writes the raw `RednotePost[]` array to disk, while stdout still prints the summary list above.

`search` stdout (both `md` and `json`):

```text
id=<database nanoid>
title=<post title>
like=<liked count>
```

`search --format json --save PATH` writes the raw `RednotePost[]` array to disk, while stdout still prints the summary list above.

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

`interact`:

```json
{
  "ok": true,
  "message": "string"
}
```

## Storage

The CLI stores browser instances and metadata under:

```text
~/.skills-router/rednote/instances
```

Inspect the current environment and resolved paths with:

```bash
rednote env
```

## Troubleshooting

If a command fails, check these in order:

- the instance name is correct
- the browser instance was created or connected
- login was completed for that instance
- the required flag such as `--keyword`, `--url`, or `--id` was provided

## Repository

- Homepage: https://github.com/skills-router/skills-store/tree/main/packages/rednote-cli
- Issues: https://github.com/skills-router/skills-store/issues

## License

MIT
