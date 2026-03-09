import { stripTypeScriptTypes } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';

const packageRoot = path.dirname(fileURLToPath(import.meta.url));
const sourceRoot = path.join(packageRoot, 'scripts');
const outputRoot = path.join(packageRoot, 'dist');

function rewriteRuntimeImports(code) {
  return code
    .replace(/(from\s*['"])(\.\.?\/[^'"]+)\.ts(['"])/g, '$1$2.js$3')
    .replace(/(import\s*\(\s*['"])(\.\.?\/[^'"]+)\.ts(['"]\s*\))/g, '$1$2.js$3');
}

function normalizeShebang(code) {
  return code.replace(/(^#!.*\n?)/, '#!/usr/bin/env node\n');
}

async function collectTypescriptFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...(await collectTypescriptFiles(fullPath)));
      continue;
    }

    if (entry.isFile() && entry.name.endsWith('.ts')) {
      files.push(fullPath);
    }
  }

  return files;
}

async function buildFile(sourceFile) {
  const relativePath = path.relative(sourceRoot, sourceFile);
  const outputFile = path.join(outputRoot, relativePath.replace(/\.ts$/u, '.js'));
  const source = await readFile(sourceFile, 'utf8');
  const transformed = stripTypeScriptTypes(source, { mode: 'transform' });
  const runtimeReady = rewriteRuntimeImports(normalizeShebang(transformed));

  await mkdir(path.dirname(outputFile), { recursive: true });
  await writeFile(outputFile, runtimeReady, 'utf8');
}

async function main() {
  await rm(outputRoot, { recursive: true, force: true });
  const sourceFiles = await collectTypescriptFiles(sourceRoot);
  await Promise.all(sourceFiles.map(buildFile));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(1);
});
