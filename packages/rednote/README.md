# rednote

A Xiaohongshu (RED) automation CLI.

## Run with npx

```bash
npx -y @skills-store/rednote browser list
npx -y @skills-store/rednote browser create --name seller-main --browser chrome
npx -y @skills-store/rednote browser connect --instance seller-main
npx -y @skills-store/rednote login --instance seller-main
npx -y @skills-store/rednote search --instance seller-main --keyword 护肤
npx -y @skills-store/rednote get-feed-detail --instance seller-main --url "https://www.xiaohongshu.com/explore/<id>?xsec_token=<token>"
npx -y @skills-store/rednote get-profile --instance seller-main --id USER_ID
```

## Install globally

```bash
npm install -g @skills-store/rednote
```

## Storage

The CLI stores custom browser instances and metadata under:

```text
~/.skills-router/rednote/instances
```

Run this to inspect the current environment and exact resolved paths:

```bash
npx -y @skills-store/rednote env
```
