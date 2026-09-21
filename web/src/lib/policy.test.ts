/// <reference types="node" />

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";
import { z } from "zod";

import { legalActions, replayMoves } from "@/lib/go-rules";
import {
  activityOf,
  chooseAction,
  dynamicsOf,
  logitsOf,
  parseGraph,
  parseManifest,
  parsePolicy,
  valueOf,
} from "@/lib/policy";

const fixtureSchema = z.object({
  cases: z.array(
    z.object({
      activity: z.array(z.number()),
      chosen_action: z.number().int(),
      logits: z.array(z.number()),
      moves: z.array(z.number().int()),
      name: z.string(),
      size: z.number().int(),
      value: z.number(),
    })
  ),
  edge_count: z.number().int(),
  graph_sha256: z.string(),
  node_count: z.number().int(),
  policies: z.record(
    z.string(),
    z.object({ seed: z.number().int(), sha256: z.string() })
  ),
  steps: z.number().int(),
});

const fixture = fixtureSchema.parse(
  JSON.parse(
    readFileSync(
      new URL("../../../shared/policy-conformance.json", import.meta.url),
      "utf-8"
    )
  )
);

const bundleRoot = new URL("../../public/flygo/", import.meta.url);

const readBytes = (name: string): Buffer =>
  readFileSync(new URL(name, bundleRoot));

const readBuffer = (name: string): ArrayBuffer => {
  const bytes = readBytes(name);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength
  ) as ArrayBuffer;
};

const manifest = parseManifest(
  JSON.parse(readBytes("manifest.json").toString("utf-8"))
);
const graph = parseGraph(readBuffer(manifest.graph.file), manifest.graph);

const maxDifference = (actual: Float32Array, expected: number[]): number => {
  let worst = 0;
  for (let index = 0; index < expected.length; index += 1) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  return worst;
};

describe("browser policy conformance with the Python engine", () => {
  it("loads the exported graph and every policy", () => {
    expect(graph.nodeCount).toBe(fixture.node_count);
    expect(graph.edgeCount).toBe(fixture.edge_count);
    expect(manifest.policies).toHaveLength(
      Object.keys(fixture.policies).length
    );
  });

  it("verifies the exported binaries against their recorded hashes", () => {
    const graphHash = createHash("sha256")
      .update(readBytes(manifest.graph.file))
      .digest("hex");
    expect(graphHash).toBe(fixture.graph_sha256);
    for (const entry of manifest.policies) {
      const recorded = fixture.policies[String(entry.size)];
      expect(recorded, `size ${entry.size}`).toBeDefined();
      const digest = createHash("sha256")
        .update(readBytes(entry.file))
        .digest("hex");
      expect(digest).toBe(recorded?.sha256);
    }
  });

  it("reproduces every recorded activity and logits vector", () => {
    let worstActivity = 0;
    let worstLogits = 0;
    for (const testCase of fixture.cases) {
      const entry = manifest.policies.find(
        (candidate) => candidate.size === testCase.size
      );
      if (!entry) {
        throw new Error(
          `The bundle has no policy for a ${testCase.size} board`
        );
      }
      const policy = parsePolicy(readBuffer(entry.file), {
        actionCount: entry.action_count,
        featureCount: entry.feature_count,
        nodeCount: manifest.graph.node_count,
        size: entry.size,
      });
      const position = replayMoves(testCase.size, testCase.moves);
      const activity = activityOf(
        { dynamics: dynamicsOf(manifest), graph, policy },
        position,
        fixture.steps
      );
      const logits = logitsOf(policy, activity);
      const value = valueOf(policy, activity);

      expect(activity).toHaveLength(testCase.activity.length);
      expect(logits).toHaveLength(testCase.logits.length);
      worstActivity = Math.max(
        worstActivity,
        maxDifference(activity, testCase.activity)
      );
      worstLogits = Math.max(
        worstLogits,
        maxDifference(logits, testCase.logits)
      );
      expect(chooseAction(logits, legalActions(position))).toBe(
        testCase.chosen_action
      );
      expect(Math.abs(value - testCase.value)).toBeLessThan(1e-5);
    }

    // Measured maximum differences against the Python engine are 4.3e-7 and
    // 4.8e-7, which is float32 rounding scale. Keep an order of magnitude of
    // headroom so the check stays stable across JavaScript engines.
    expect(worstActivity).toBeLessThan(1e-5);
    expect(worstLogits).toBeLessThan(1e-5);
  });
});
