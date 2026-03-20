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
rednote login
rednote status
rednote search --keyword 护肤
rednote interact --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy" --like --collect --comment "写得真好"
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
rednote status
```

Use `status` to confirm whether an instance exists, is running, and appears logged in.

### `check-login`

```bash
rednote check-login
```

Use `check-login` when you only want to verify whether the session is still valid.

### `login`

```bash
rednote login
```

Use `login` after `browser connect` if the instance is not authenticated yet.

### `home`

```bash
rednote home --format md --save
```

Use `home` when you want the current home feed and optionally want to save it to disk.

The terminal output always uses the compact summary format below, even when `--format json` is selected:

```text
id=<database nanoid>
title=<post title>
like=<liked count>

id=...
title=...
like=...
```

Captured home feed posts are upserted into `~/.skills-router/rednote/main.db` (the same path returned by `rednote env`). They are stored in the `rednote_posts` table, and the printed `id` is that table's `nanoid(16)` primary key.

### `search`

```bash
rednote search --keyword 护肤
rednote search --keyword 护肤 --format json --save ./output/search.jsonl
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
rednote get-feed-detail --id <nanoid>
rednote get-feed-detail --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

Use `get-feed-detail` when you already have a Xiaohongshu note URL, or when you have a database `id` returned by `home` or `search`. With `--id`, the CLI looks up the saved URL from `~/.skills-router/rednote/main.db` and then navigates with that raw URL.

Captured note details and comments are also upserted into `~/.skills-router/rednote/main.db` in `rednote_post_details` and `rednote_post_comments`.

### `get-profile`

```bash
rednote get-profile --id USER_ID
```

Use `get-profile` when you want author or account profile information.


### `interact`

```bash
rednote interact --id <nanoid> --like --collect
rednote interact --id <nanoid> --like --collect --comment "写得真好"
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
