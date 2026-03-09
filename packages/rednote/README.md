# @skills-store/rednote

A Xiaohongshu (RED) automation CLI for browser session management, login, search, feed detail lookup, and profile lookup.

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
6. home, search, get-feed-detail, or get-profile
```

## Quick start

```bash
rednote env
rednote browser create --name seller-main --browser chrome --port 9222
rednote browser connect --instance seller-main
rednote login --instance seller-main
rednote status --instance seller-main
rednote search --instance seller-main --keyword 护肤
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

### `search`

```bash
rednote search --instance seller-main --keyword 护肤
rednote search --instance seller-main --keyword 护肤 --format json --save ./output/search.jsonl
```

Use `search` for keyword-based note lookup.

### `get-feed-detail`

```bash
rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/xxx?xsec_token=yyy"
```

Use `get-feed-detail` when you already have a Xiaohongshu note URL.

### `get-profile`

```bash
rednote get-profile --instance seller-main --id USER_ID
```

Use `get-profile` when you want author or account profile information.

## Important flags

- `--instance NAME` selects the browser instance for account-scoped commands.
- `--format json` is best for scripting.
- `--format md` is best for direct reading.
- `--save` is useful for `home` and `search` when you want saved output.
- `--keyword` is required for `search`.
- `--url` is required for `get-feed-detail`.
- `--id` is required for `get-profile`.

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

- Homepage: https://github.com/skills-router/skills-store/tree/main/packages/rednote
- Issues: https://github.com/skills-router/skills-store/issues

## License

MIT
