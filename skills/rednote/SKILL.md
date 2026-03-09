---
name: rednote
description: Use when the user needs to publish, search, query, or otherwise operate Xiaohongshu (RedNote).
---

# Rednote Commands

Use this skill only for the CLI command surface of `@skills-store/rednote` when the user wants to publish, search, query, or operate Xiaohongshu from the terminal.

Focus on telling the user which command to run, which flags matter, and what order to use the commands in.

## Preferred command style

Prefer global-install examples first:

```bash
npm install -g @skills-store/rednote
bun add -g @skills-store/rednote
rednote <command> [...args]
```

Only mention `npx -y @skills-store/rednote ...` if the user explicitly asks for one-off execution without global installation.

Only show local repo commands if the user is explicitly developing the CLI.

## Command order

For most operational tasks, use this sequence:

1. `env`
2. `browser list` or `browser create`
3. `browser connect`
4. `login` or `check-login`
5. `status`
6. `publish`, `home`, `search`, `get-feed-detail`, or `get-profile`

## Quick reference

### `browser`

List browser instances:

```bash
rednote browser list
```

Create a browser instance:

```bash
rednote browser create --name seller-main --browser chrome --port 9222
```

Connect to an instance:

```bash
rednote browser connect --instance seller-main
```

Connect with explicit profile path:

```bash
rednote browser connect --browser edge --user-data-dir /tmp/edge-profile --port 9223
```

Remove an instance:

```bash
rednote browser remove --name seller-main
```

Use `browser` whenever the user needs browser setup, profile management, or a reusable instance for later commands.

### `env`

Show runtime and storage info:

```bash
rednote env
rednote env --format json
```

Use `env` first when the user is debugging installation or local setup.

### `status`

Check the selected instance state:

```bash
rednote status --instance seller-main
```

Use `status` to confirm whether the instance exists, is running, and appears logged in.

### `check-login`

Check login state only:

```bash
rednote check-login --instance seller-main
```

Use this when the user specifically wants to know whether the account session is still valid.

### `login`

Open login flow for an instance:

```bash
rednote login --instance seller-main
```

Use `login` after `browser connect` if the account is not authenticated yet.

### `publish`

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

Parameter guidance:

- `--instance NAME` is optional and defaults to the saved last connected instance.
- `--type video|image|article` is optional; if omitted, the CLI infers it from `--video` or `--image`, otherwise it falls back to `article`.
- `--title TEXT` is required for every publish type.
- `--content TEXT` is required for every publish type; for video and image it is the description, for article it is Markdown content.
- `--video PATH` is required for `video` and only one file is allowed.
- `--image PATH` is required for `image`; repeat it for multiple images, up to 15 files, and the first image becomes the cover.
- `--tag TEXT` is optional and repeatable for `video` and `image`; tags are normalized and deduplicated.
- `--publish` publishes immediately; without it, `publish` saves a draft by default.
- `article` must not be combined with `--video`, `--image`, or `--tag`.
- `video` and `image` must not be mixed in the same command.

Use `publish` when the user wants to post or save drafts to Xiaohongshu from an authenticated browser instance.

### `home`

Read home feed content:

```bash
rednote home --instance seller-main --format md --save
```

Use `home` when the user wants the current home feed or wants to save it to disk.

### `search`

Search by keyword:

```bash
rednote search --instance seller-main --keyword 护肤
rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
```

Use `search` when the user wants RED notes by keyword and optionally wants machine-readable output.

### `get-feed-detail`

Fetch a note by URL:

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

Use this when the user already has a Xiaohongshu note URL and wants structured detail data.

### `get-profile`

Fetch a profile by user ID:

```bash
rednote get-profile --instance seller-main --id USER_ID
```

Use this when the user wants author or account profile information.

## Flag guidance

- `--instance NAME` picks the browser instance for account-scoped commands.
- `--format json` is best for scripting.
- `--format md` is best for direct reading.
- `--save` is useful for `home` and `search` when the user wants saved output.
- `--keyword` is required for `search`.
- `--type`, `--title`, `--content`, `--video`, `--image`, `--tag`, and `--publish` are the main `publish` flags.
- `publish` usually requires a connected and logged-in instance before running; without `--publish`, it saves a draft.
- `--url` is required for `get-feed-detail`.
- `--id` is required for `get-profile`.

## Response style

When answering users:

- lead with the exact command they should run
- include only the flags needed for that task
- prefer one happy-path example first
- mention `browser connect` and `login` if the command requires an authenticated instance

## Troubleshooting

If a command fails, check these in order:

- the instance name is correct
- the browser instance was created or connected
- login was completed for that instance
- the user provided the required flag such as `--keyword`, `--url`, or `--id`
