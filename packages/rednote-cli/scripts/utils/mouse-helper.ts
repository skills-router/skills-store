import { createCursor, getRandomPagePoint } from 'playwright-ghost-cursor';
import type { Locator, Page } from 'playwright-core';

type MouseMoveOptions = {
  locator?: Locator | null;
  settleMs?: number;
  moveSpeed?: number;
};

type MouseWheelOptions = MouseMoveOptions & {
  deltaY?: number;
  repeats?: number;
  stepPauseMs?: number;
  moveBeforeScroll?: boolean;
};

type MousePresenceOptions = {
  locator?: Locator | null;
  minDurationMs?: number;
  maxDurationMs?: number;
  allowScroll?: boolean;
};

type GhostCursorHandle = ReturnType<typeof createCursor>;

type MousePoint = {
  x: number;
  y: number;
};

const ghostCursorCache = new WeakMap<Page, GhostCursorHandle>();

function randomInt(min: number, max: number) {
  const lower = Math.min(min, max);
  const upper = Math.max(min, max);
  return Math.floor(Math.random() * (upper - lower + 1)) + lower;
}

function getCursor(page: Page) {
  const cached = ghostCursorCache.get(page);
  if (cached) {
    return cached;
  }

  const cursor = createCursor(page as never);
  ghostCursorCache.set(page, cursor);
  return cursor;
}

async function resolveRandomPoint(page: Page): Promise<MousePoint> {
  const point = await getRandomPagePoint(page as never).catch(() => null);
  if (point) {
    return point;
  }

  const viewport = page.viewportSize() ?? { width: 1280, height: 720 };
  return {
    x: Math.round(viewport.width * 0.5),
    y: Math.round(viewport.height * 0.35),
  };
}

export async function simulateMouseMove(page: Page, options: MouseMoveOptions = {}) {
  const cursor = getCursor(page);
  const settleMs = options.settleMs ?? 120;
  const moveSpeed = options.moveSpeed ?? randomInt(6, 14);

  if (options.locator) {
    const handle = await options.locator.elementHandle().catch(() => null);
    if (handle) {
      await cursor.move(handle as never, {
        moveSpeed,
        moveDelay: 0,
        paddingPercentage: 70,
      });
      if (settleMs > 0) {
        await page.waitForTimeout(settleMs);
      }
      return cursor.getLocation();
    }
  }

  const point = await resolveRandomPoint(page);
  await cursor.moveTo(point, {
    moveSpeed,
    moveDelay: 0,
  });

  if (settleMs > 0) {
    await page.waitForTimeout(settleMs);
  }

  return point;
}

export async function simulateMouseWheel(page: Page, options: MouseWheelOptions = {}) {
  if (options.moveBeforeScroll !== false) {
    await simulateMouseMove(page, options);
  }

  const repeats = Math.max(options.repeats ?? 1, 1);
  const deltaY = options.deltaY ?? 360;
  const stepPauseMs = options.stepPauseMs ?? 180;
  const settleMs = options.settleMs ?? 150;

  for (let index = 0; index < repeats; index += 1) {
    await page.mouse.wheel(0, deltaY);
    const pause = index === repeats - 1 ? settleMs : stepPauseMs;
    if (pause > 0) {
      await page.waitForTimeout(pause);
    }
  }
}

export async function simulateMousePresence(page: Page, options: MousePresenceOptions = {}) {
  const cursor = getCursor(page);
  const duration = randomInt(options.minDurationMs ?? 3_000, options.maxDurationMs ?? 5_000);
  const deadline = Date.now() + duration;
  let movedToLocator = false;

  while (Date.now() < deadline) {
    if (options.locator && !movedToLocator) {
      await simulateMouseMove(page, {
        locator: options.locator,
        settleMs: randomInt(80, 220),
        moveSpeed: randomInt(7, 15),
      }).catch(() => {});
      movedToLocator = true;
    } else {
      const point = await resolveRandomPoint(page);
      await cursor.moveTo(point, {
        moveSpeed: randomInt(5, 14),
        moveDelay: randomInt(0, 120),
        randomizeMoveDelay: true,
      }).catch(() => {});
    }

    if (options.allowScroll && Math.random() < 0.2) {
      await page.mouse.wheel(0, randomInt(-160, 220)).catch(() => {});
    }

    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      break;
    }

    await page.waitForTimeout(Math.min(randomInt(180, 650), remaining));
  }
}
