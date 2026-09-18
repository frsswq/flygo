/// <reference types="node" />

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { legalActions, replayMoves } from "@/lib/go-rules";
import { searchPosition } from "@/lib/mcts";
import {
  dynamicsOf,
  parseGraph,
  parseManifest,
  parsePolicy,
} from "@/lib/policy";

const root = new URL("../../public/flygo/", import.meta.url);

const bufferOf = (name: string): ArrayBuffer => {
  const bytes = readFileSync(new URL(name, root));
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength
  ) as ArrayBuffer;
};

const manifest = parseManifest(
  JSON.parse(readFileSync(new URL("manifest.json", root), "utf-8"))
);
const graph = parseGraph(bufferOf(manifest.graph.file));
const entry = manifest.policies.find((policy) => policy.size === 5);
if (!entry) {
  throw new Error("The fixture bundle has no 5x5 policy");
}
const policy = parsePolicy(bufferOf(entry.file));

describe("browser MCTS", () => {
  it("returns a visited legal action with a fixed simulation budget", () => {
    const position = replayMoves(5, []);

    const result = searchPosition(
      graph,
      policy,
      dynamicsOf(manifest),
      position,
      { maxSimulations: 8 }
    );

    expect(legalActions(position)).toContain(result.action);
    expect({
      childVisits: Object.values(result.visits).reduce(
        (sum, visits) => sum + visits,
        0
      ),
      simulations: result.simulations,
    }).toStrictEqual({ childVisits: 7, simulations: 8 });
    expect(result.visits[result.action]).toBeGreaterThan(0);
    expect(result.activity).toHaveLength(graph.nodeCount);
    expect(Math.abs(result.rootValue)).toBeLessThanOrEqual(1);
  });

  it("rejects a finished game", () => {
    const finished = replayMoves(5, [25, 25]);

    expect(() =>
      searchPosition(graph, policy, dynamicsOf(manifest), finished, {
        consecutivePasses: 2,
        maxSimulations: 1,
      })
    ).toThrow(/finished/u);
  });
});
