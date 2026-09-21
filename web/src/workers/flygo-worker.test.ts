/// <reference types="node" />

import { once } from "node:events";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

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

const posted: FlyGoWorkerResponse[] = [];
const listeners: ((event: { data: FlyGoWorkerRequest }) => void)[] = [];

const send = async (
  request: FlyGoWorkerRequest
): Promise<FlyGoWorkerResponse> => {
  const before = posted.length;
  for (const listener of listeners) {
    listener({ data: request });
  }
  await vi.waitFor(() => {
    expect(posted.length).toBeGreaterThan(before);
  });
  const response = posted.at(-1);
  if (!response) {
    throw new Error("The worker did not answer");
  }
  return response;
};

describe("flygo worker bundle loading", () => {
  let baseUrl = "";

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
    await import("@/workers/flygo-worker");
    server = createServer((request, response) => {
      const body = current.get(request.url ?? "");
      if (!body) {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, { "content-type": "application/octet-stream" });
      response.end(body);
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address() as AddressInfo;
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  afterAll(() => {
    server.close();
    vi.unstubAllGlobals();
  });

  it("answers a published bundle with real graph activity", async () => {
    current = new Map(pristine);

    const response = await send({
      baseUrl,
      requestId: 1,
      size: 5,
      type: "initialize",
    });

    if (response.type !== "ready") {
      throw new Error(`The worker answered with ${response.type}`);
    }
    expect(response.activity).toHaveLength(461);
    expect([...response.activity].every(Number.isFinite)).toBeTruthy();
  });

  // A distinct board size keeps this case on its own worker cache key.
  it("answers with an error instead of a loaded malformed bundle", async () => {
    const manifest = JSON.parse(String(pristine.get(MANIFEST))) as {
      graph: { sha256: string };
    };
    manifest.graph.sha256 = "0".repeat(64);
    current = new Map(pristine).set(
      MANIFEST,
      Buffer.from(JSON.stringify(manifest))
    );

    const response = await send({
      baseUrl,
      requestId: 2,
      size: 19,
      type: "initialize",
    });

    if (response.type !== "error") {
      throw new Error(`The worker answered with ${response.type}`);
    }
    expect(response.message).toMatch(
      /graph\.bin does not match its manifest sha256/u
    );
  });
});
