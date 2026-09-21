/// <reference types="node" />

import { createHash } from "node:crypto";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  loadWebBundle,
  parseGraph,
  parseManifest,
  parsePolicy,
} from "@/lib/policy";

const bundleRoot = new URL("../../public/flygo/", import.meta.url);

const bytesOf = (name: string): Buffer =>
  readFileSync(new URL(name, bundleRoot));

const MANIFEST = "/manifest.json";
const GRAPH = "/graph.bin";
const policyPath = (size: number): string => `/policy-${size}.bin`;

const pristine = new Map<string, Buffer>([
  [MANIFEST, bytesOf("manifest.json")],
  [GRAPH, bytesOf("graph.bin")],
  [policyPath(5), bytesOf("policy-5.bin")],
  [policyPath(19), bytesOf("policy-19.bin")],
]);

interface FixtureManifest {
  graph: {
    edge_count: number;
    file: string;
    node_count: number;
    sha256: string;
  };
  policies: {
    action_count: number;
    feature_count: number;
    file: string;
    seed: number;
    sha256: string;
    size: number;
  }[];
  ruleset: string;
}

const sha256 = (bytes: Buffer): string =>
  createHash("sha256").update(bytes).digest("hex");

const manifestOf = (): FixtureManifest =>
  JSON.parse(String(pristine.get(MANIFEST))) as FixtureManifest;

const bufferOf = (bytes: Buffer): ArrayBuffer =>
  bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength
  ) as ArrayBuffer;

/** Serve bundle bytes, replacing the entries each test changes. */
let current = new Map(pristine);
let server: Server;

const replaceGraph = (mutate: (bytes: Buffer) => void): void => {
  const graph = Buffer.from(pristine.get(GRAPH) as Buffer);
  mutate(graph);
  const manifest = manifestOf();
  manifest.graph.sha256 = sha256(graph);
  current = new Map(pristine)
    .set(GRAPH, graph)
    .set(MANIFEST, Buffer.from(JSON.stringify(manifest)));
};

const replacePolicy = (size: number, mutate: (bytes: Buffer) => void): void => {
  const policy = Buffer.from(pristine.get(policyPath(size)) as Buffer);
  mutate(policy);
  const manifest = manifestOf();
  const entry = manifest.policies.find((candidate) => candidate.size === size);
  if (!entry) {
    throw new Error(`No fixture policy for size ${size}`);
  }
  entry.sha256 = sha256(policy);
  current = new Map(pristine)
    .set(policyPath(size), policy)
    .set(MANIFEST, Buffer.from(JSON.stringify(manifest)));
};

describe("bundle integrity over http", () => {
  let baseUrl = "";

  beforeAll(async () => {
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
    baseUrl = `http://127.0.0.1:${address.port}/`;
  });

  afterAll(() => {
    server.close();
  });

  it("loads a valid published bundle", async () => {
    current = new Map(pristine);

    const bundle = await loadWebBundle(baseUrl, 5);

    expect(bundle.graph.nodeCount).toBe(461);
    expect(bundle.policies.get(5)?.size).toBe(5);
    expect(bundle.ruleset).toBe(
      "Tromp-Taylor, area scoring, positional superko"
    );
  });

  it("rejects a graph whose hash does not match the manifest", async () => {
    const manifest = manifestOf();
    manifest.graph.sha256 = "0".repeat(64);
    current = new Map(pristine).set(
      MANIFEST,
      Buffer.from(JSON.stringify(manifest))
    );

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /graph\.bin does not match its manifest sha256/u
    );
  });

  it("rejects a policy whose hash does not match the manifest", async () => {
    const manifest = manifestOf();
    for (const entry of manifest.policies) {
      entry.sha256 = "0".repeat(64);
    }
    current = new Map(pristine).set(
      MANIFEST,
      Buffer.from(JSON.stringify(manifest))
    );

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /policy-5\.bin does not match its manifest sha256/u
    );
  });

  it("rejects non-finite policy weights even with a matching hash", async () => {
    replacePolicy(5, (bytes) => {
      bytes.writeFloatLE(Number.NaN, 24);
    });

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /Policy bundle holds non-finite weights/u
    );
  });

  it("rejects non-finite graph weights even with a matching hash", async () => {
    replaceGraph((bytes) => {
      const nodeCount = bytes.readUInt32LE(8);
      const edgeCount = bytes.readUInt32LE(12);
      bytes.writeFloatLE(
        Number.POSITIVE_INFINITY,
        16 + nodeCount * 4 + edgeCount * 8
      );
    });

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /Graph bundle holds non-finite numbers/u
    );
  });

  it("rejects an out-of-range edge index even with a matching hash", async () => {
    replaceGraph((bytes) => {
      const nodeCount = bytes.readUInt32LE(8);
      bytes.writeUInt32LE(nodeCount + 5, 16 + nodeCount * 4);
    });

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /Graph bundle holds an out-of-range edge index/u
    );
  });

  it("rejects a zero incoming strength even with a matching hash", async () => {
    replaceGraph((bytes) => {
      const nodeCount = bytes.readUInt32LE(8);
      const edgeCount = bytes.readUInt32LE(12);
      bytes.writeFloatLE(0, 16 + nodeCount * 4 + edgeCount * 12);
    });

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /Graph bundle holds a zero incoming strength/u
    );
  });

  it("rejects a graph whose header disagrees with the manifest", async () => {
    const manifest = manifestOf();
    manifest.graph.node_count += 1;
    manifest.graph.sha256 = sha256(pristine.get(GRAPH) as Buffer);
    current = new Map(pristine).set(
      MANIFEST,
      Buffer.from(JSON.stringify(manifest))
    );

    await expect(loadWebBundle(baseUrl, 5)).rejects.toThrow(
      /Graph bundle header does not match the manifest/u
    );
  });
});

describe("bundle integrity in the manifest schema", () => {
  it("rejects a manifest without policy hashes", () => {
    const manifest = manifestOf();
    const [entry] = manifest.policies;
    const withoutHash = { ...entry, sha256: undefined };

    expect(() =>
      parseManifest({ ...manifest, policies: [withoutHash] })
    ).toThrow(/sha256/u);
  });

  it("rejects a manifest whose digest is not a sha256 string", () => {
    expect(() =>
      parseManifest({
        ...manifestOf(),
        graph: { ...manifestOf().graph, sha256: "not-a-digest" },
      })
    ).toThrow(/sha256/u);
  });

  it("rejects a policy whose node count disagrees with the graph", () => {
    const manifest = parseManifest(manifestOf());
    const entry = manifest.policies.find((policy) => policy.size === 5);
    if (!entry) {
      throw new Error("The fixture bundle has no 5x5 policy");
    }

    expect(() =>
      parsePolicy(bufferOf(pristine.get(policyPath(5)) as Buffer), {
        actionCount: entry.action_count,
        featureCount: entry.feature_count,
        nodeCount: manifest.graph.node_count + 1,
        size: 5,
      })
    ).toThrow(/Policy bundle node count does not match the graph/u);
  });

  it("rejects a policy whose board size disagrees with its header", () => {
    const manifest = parseManifest(manifestOf());
    const entry = manifest.policies.find((policy) => policy.size === 5);
    if (!entry) {
      throw new Error("The fixture bundle has no 5x5 policy");
    }

    expect(() =>
      parsePolicy(bufferOf(pristine.get(policyPath(5)) as Buffer), {
        actionCount: entry.action_count,
        featureCount: entry.feature_count,
        nodeCount: manifest.graph.node_count,
        size: 19,
      })
    ).toThrow(/Policy bundle header does not match the manifest/u);
  });

  it("parses a graph that matches its manifest identity", () => {
    const manifest = parseManifest(manifestOf());

    const graph = parseGraph(bufferOf(pristine.get(GRAPH) as Buffer), {
      edge_count: manifest.graph.edge_count,
      node_count: manifest.graph.node_count,
    });

    expect(graph.nodeCount).toBe(manifest.graph.node_count);
    expect(graph.edgeCount).toBe(manifest.graph.edge_count);
  });
});
