/// <reference types="node" />

import { once } from "node:events";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import type {
  FlyGoWorkerRequest,
  FlyGoWorkerResponse,
} from "@/lib/worker-protocol";

const bundleRoot = new URL("../../public/flygo/", import.meta.url);

const bytesOf = (name: string): Buffer =>
  readFileSync(new URL(name, bundleRoot));

const MANIFEST = "/manifest.json";
const GRAPH = "/graph.bin";

const pristine = new Map<string, Buffer>([
  [MANIFEST, bytesOf("manifest.json")],
  [GRAPH, bytesOf("graph.bin")],
  ["/policy-5.bin", bytesOf("policy-5.bin")],
  ["/policy-19.bin", bytesOf("policy-19.bin")],
]);

/** Serve bundle bytes, replacing the entries each test changes. */
let current = new Map(pristine);
let server: Server;
let baseUrl = "";

/** Paths that fail their next request, so one transient fault can be injected. */
const failOnce = new Set<string>();
/** Paths whose next request is delayed, so an in-flight load can be observed. */
const delayOnce = new Map<string, number>();
const requestCounts = new Map<string, number>();

const posted: FlyGoWorkerResponse[] = [];
const listeners: ((event: { data: FlyGoWorkerRequest }) => void)[] = [];

const dispatch = (request: FlyGoWorkerRequest): void => {
  for (const listener of listeners) {
    listener({ data: request });
  }
};

const answerAfter = async (from: number): Promise<FlyGoWorkerResponse> => {
  await vi.waitFor(() => {
    expect(posted.length).toBeGreaterThan(from);
  });
  const response = posted.at(-1);
  if (!response) {
    throw new Error("The worker did not answer");
  }
  return response;
};

const send = (request: FlyGoWorkerRequest): Promise<FlyGoWorkerResponse> => {
  const from = posted.length;
  dispatch(request);
  return answerAfter(from);
};

const initialize = (requestId: number, size: number): FlyGoWorkerRequest => ({
  baseUrl,
  requestId,
  size,
  type: "initialize",
});

/** Load the worker module again so each test owns a fresh bundle cache. */
const startWorker = async (): Promise<void> => {
  vi.resetModules();
  listeners.length = 0;
  await import("@/workers/flygo-worker");
};

describe("flygo worker bundle loading", () => {
  beforeAll(async () => {
    vi.stubGlobal("self", {
      addEventListener: (
        type: string,
        listener: (event: { data: FlyGoWorkerRequest }) => void
      ) => {
        if (type === "message") {
          listeners.push(listener);
        }
      },
      postMessage: (response: FlyGoWorkerResponse) => {
        posted.push(response);
      },
    });
    server = createServer((request, response) => {
      const path = request.url ?? "";
      requestCounts.set(path, (requestCounts.get(path) ?? 0) + 1);
      if (failOnce.delete(path)) {
        response.writeHead(503).end();
        return;
      }
      const body = current.get(path);
      if (!body) {
        response.writeHead(404).end();
        return;
      }
      const answer = (): void => {
        response.writeHead(200, { "content-type": "application/octet-stream" });
        response.end(body);
      };
      const delay = delayOnce.get(path);
      if (delay === undefined) {
        answer();
        return;
      }
      delayOnce.delete(path);
      setTimeout(answer, delay);
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  beforeEach(async () => {
    current = new Map(pristine);
    failOnce.clear();
    delayOnce.clear();
    requestCounts.clear();
    posted.length = 0;
    await startWorker();
  });

  afterAll(() => {
    server.close();
    vi.unstubAllGlobals();
  });

  it("answers a published bundle with real graph activity", async () => {
    const response = await send(initialize(1, 5));

    if (response.type !== "ready") {
      throw new Error(`The worker answered with ${response.type}`);
    }
    expect(response.activity).toHaveLength(461);
    expect([...response.activity].every(Number.isFinite)).toBeTruthy();
  });

  it("answers with an error instead of a loaded malformed bundle", async () => {
    const manifest = JSON.parse(String(pristine.get(MANIFEST))) as {
      graph: { sha256: string };
    };
    manifest.graph.sha256 = "0".repeat(64);
    current = new Map(pristine).set(
      MANIFEST,
      Buffer.from(JSON.stringify(manifest))
    );

    const response = await send(initialize(1, 5));

    if (response.type !== "error") {
      throw new Error(`The worker answered with ${response.type}`);
    }
    expect(response.message).toMatch(
      /graph\.bin does not match its manifest sha256/u
    );
  });

  it("recovers on the same worker after a transient manifest failure", async () => {
    failOnce.add(MANIFEST);

    const failed = await send(initialize(1, 5));
    const recovered = await send(initialize(2, 5));

    if (failed.type !== "error") {
      throw new Error(`The first load answered with ${failed.type}`);
    }
    expect(failed.message).toMatch(/manifest failed with status 503/u);
    expect(recovered.type).toBe("ready");
  });

  it("recovers on the same worker after a transient asset failure", async () => {
    failOnce.add(GRAPH);

    const failed = await send(initialize(1, 5));
    const recovered = await send(initialize(2, 5));

    if (failed.type !== "error") {
      throw new Error(`The first load answered with ${failed.type}`);
    }
    expect(failed.message).toMatch(/Bundle files failed to load/u);
    expect(recovered.type).toBe("ready");
  });

  it("shares one in-flight load between concurrent requests", async () => {
    delayOnce.set(MANIFEST, 200);
    const from = posted.length;

    dispatch(initialize(1, 5));
    dispatch(initialize(2, 5));
    await vi.waitFor(() => {
      expect(posted.length).toBeGreaterThanOrEqual(from + 2);
    });

    const answers = posted.slice(from);
    expect(answers.map((answer) => answer.type)).toStrictEqual([
      "ready",
      "ready",
    ]);
    expect(requestCounts.get(MANIFEST)).toBe(1);
  });

  it("keeps a newer cache entry when an older load rejects", async () => {
    delayOnce.set(MANIFEST, 300);
    failOnce.add(MANIFEST);
    const from = posted.length;

    dispatch(initialize(1, 5));
    await vi.waitFor(() => {
      expect(requestCounts.get(MANIFEST)).toBe(1);
    });
    dispatch(initialize(2, 5));
    await vi.waitFor(() => {
      expect(posted.length).toBeGreaterThanOrEqual(from + 2);
    });

    const cached = await send(initialize(3, 5));

    expect(requestCounts.get(MANIFEST)).toBe(2);
    expect(cached.type).toBe("ready");
  });
});
