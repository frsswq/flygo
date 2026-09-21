/// <reference types="node" />

import { execFileSync } from "node:child_process";
import { once } from "node:events";
import { existsSync, mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { homedir, tmpdir } from "node:os";
import path from "node:path";

import type { Browser, Page } from "playwright-core";
import { chromium } from "playwright-core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const webRoot = path.resolve(import.meta.dirname, "..");
const bundleRoot = path.join(webRoot, "public", "flygo");

/** How long the first bundle manifest request hangs, in milliseconds. */
const LOAD_DELAY_MS = 1500;
/** How long the UI is given to settle after pending work is released. */
const SETTLE_MS = 4000;

const browserCandidates = (): string[] => {
  const candidates: string[] = [];
  if (process.env.BROWSER_BIN) {
    candidates.push(process.env.BROWSER_BIN);
  }
  const cache = path.join(homedir(), ".cache", "ms-playwright");
  if (existsSync(cache)) {
    for (const entry of readdirSync(cache).toSorted().toReversed()) {
      if (entry.startsWith("chromium-")) {
        candidates.push(
          path.join(cache, entry, "chrome-linux64", "chrome"),
          path.join(cache, entry, "chrome-linux", "chrome")
        );
      }
    }
  }
  candidates.push(
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  );
  return candidates;
};

const findBrowser = (): string => {
  const found = browserCandidates().find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error(
      "No Chrome or Chromium binary found; set BROWSER_BIN to run the browser regression"
    );
  }
  return found;
};

interface UiState {
  readonly enabledPoints: number;
  readonly status: string;
}

const contentType = (name: string): string => {
  const extension = name.slice(name.lastIndexOf("."));
  const types: Record<string, string> = {
    ".bin": "application/octet-stream",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
  };
  return types[extension] ?? "application/octet-stream";
};

interface Harness {
  readonly origin: string;
  readonly page: Page;
}

const stateOf = (page: Page): Promise<UiState> =>
  page.evaluate(() => {
    const status = document.querySelector("p[aria-live]")?.textContent ?? "";
    const points = [...document.querySelectorAll("fieldset button")];
    return {
      enabledPoints: points.filter((point) => !point.disabled).length,
      status,
    };
  });

/** Wait until the UI stops reporting that it is still loading. */
const settle = async (page: Page): Promise<UiState> => {
  try {
    await page.waitForFunction(
      () =>
        !(document.querySelector("p[aria-live]")?.textContent ?? "").includes(
          "Thinking"
        ),
      undefined,
      { timeout: SETTLE_MS }
    );
  } catch {
    // The UI stayed in a loading state; the caller asserts the reported state.
  }
  return stateOf(page);
};

const clickButton = (page: Page, text: string): Promise<void> =>
  page.getByRole("button", { exact: true, name: text }).click();

/** Click whichever label the self-play toggle currently shows. */
const toggleSelfPlay = (page: Page): Promise<void> =>
  page.getByRole("button", { name: /^(?:Self-play|Stop)$/u }).click();

describe("self-play during bundle loading", () => {
  let browser: Browser;
  let server: Server;
  let origin = "";
  let appDirectory = "";

  beforeAll(async () => {
    appDirectory = mkdtempSync(path.join(tmpdir(), "flygo-browser-"));
    execFileSync("npm", ["run", "build"], {
      cwd: webRoot,
      env: {
        ...process.env,
        FLYGO_OUT_DIR: appDirectory,
        FLYGO_PUBLIC_BASE: "/",
      },
      stdio: "pipe",
    });
    let manifestRequests = 0;
    server = createServer((request, response) => {
      const [urlPath = "/"] = (request.url ?? "/").split("?");
      const answer = (target: string, cache = false): void => {
        const body = readFileSync(target);
        response.writeHead(200, {
          "cache-control": cache ? "public, max-age=60" : "no-store",
          "content-type": contentType(target),
        });
        response.end(body);
      };
      if (urlPath.startsWith("/flygo/")) {
        const name = urlPath.slice("/flygo/".length);
        if (name === "manifest.json") {
          manifestRequests += 1;
          if (manifestRequests === 1) {
            setTimeout(
              () => answer(path.join(bundleRoot, name)),
              LOAD_DELAY_MS
            );
            return;
          }
        }
        answer(path.join(bundleRoot, name), true);
        return;
      }
      const target = path.join(appDirectory, urlPath.slice(1) || "index.html");
      if (existsSync(target)) {
        answer(target);
        return;
      }
      answer(path.join(appDirectory, "index.html"));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address() as AddressInfo;
    origin = `http://127.0.0.1:${address.port}/`;
    browser = await chromium.launch({
      executablePath: findBrowser(),
    });
  });

  afterAll(async () => {
    await browser?.close();
    server?.close();
  });

  const openHarness = async (): Promise<Harness> => {
    const page = await browser.newPage();
    await page.goto(origin);
    return { origin, page };
  };

  it("keeps a stopped self-play game playable after a slow load", async () => {
    const { page } = await openHarness();
    const initial = await stateOf(page);
    expect(initial.status).toContain("Thinking");

    await clickButton(page, "Self-play");
    await page.waitForTimeout(50);
    await clickButton(page, "Stop");

    const settled = await settle(page);

    expect(settled.status).toContain("Your turn");
    expect(settled.enabledPoints).toBeGreaterThan(0);
    await page.close();
  });

  it("survives rapid self-play toggles during the load", async () => {
    const { page } = await openHarness();

    await toggleSelfPlay(page);
    await toggleSelfPlay(page);
    await toggleSelfPlay(page);
    await toggleSelfPlay(page);
    await toggleSelfPlay(page);
    await toggleSelfPlay(page);

    const settled = await settle(page);

    expect(settled.status).toContain("Your turn");
    expect(settled.enabledPoints).toBeGreaterThan(0);
    await page.close();
  });

  it("survives a reset during the load", async () => {
    const { page } = await openHarness();

    await clickButton(page, "New");
    const settled = await settle(page);

    expect(settled.status).toBe("Your turn");
    expect(settled.enabledPoints).toBe(361);
    await page.close();
  });

  it("survives a board size change during a loading self-play game", async () => {
    const { page } = await openHarness();

    await clickButton(page, "Self-play");
    await page.selectOption("select[aria-label='Board size']", "5");
    await clickButton(page, "Stop");

    const settled = await settle(page);

    expect(settled.status).toContain("Your turn");
    expect(settled.enabledPoints).toBeGreaterThan(0);
    await page.close();
  });
});
