/// <reference lib="webworker" />

import { replayMoves, trailingPasses } from "@/lib/go-rules";
import { searchPosition } from "@/lib/mcts";
import { activityOf, loadWebBundle } from "@/lib/policy";
import type { WebBundle } from "@/lib/policy";
import type {
  FlyGoWorkerRequest,
  FlyGoWorkerResponse,
} from "@/lib/worker-protocol";

const bundles = new Map<string, Promise<WebBundle>>();

const bundleOf = (baseUrl: string, size: number): Promise<WebBundle> => {
  const key = `${baseUrl}:${size}`;
  const cached = bundles.get(key);
  if (cached) {
    return cached;
  }
  const loading = loadWebBundle(baseUrl, size);
  bundles.set(key, loading);
  return loading;
};

const send = (response: FlyGoWorkerResponse): void => {
  self.postMessage(response, []);
};

const handle = async (request: FlyGoWorkerRequest): Promise<void> => {
  try {
    const bundle = await bundleOf(request.baseUrl, request.size);
    const policy = bundle.policies.get(request.size);
    if (!policy) {
      throw new RangeError(
        `No ${request.size}x${request.size} policy is loaded`
      );
    }
    if (request.type === "initialize") {
      const position = replayMoves(request.size, []);
      send({
        activity: activityOf(bundle.graph, policy, bundle.dynamics, position),
        requestId: request.requestId,
        type: "ready",
      });
      return;
    }
    const position = replayMoves(request.size, request.moves);
    const reading = searchPosition(
      bundle.graph,
      policy,
      bundle.dynamics,
      position,
      {
        consecutivePasses: trailingPasses(request.moves, request.size),
        timeMs: request.timeMs,
      }
    );
    send({
      action: reading.action,
      activity: reading.activity,
      elapsedMs: reading.elapsedMs,
      requestId: request.requestId,
      simulations: reading.simulations,
      type: "move",
    });
  } catch (error: unknown) {
    send({
      message: error instanceof Error ? error.message : "Unknown worker error",
      requestId: request.requestId,
      type: "error",
    });
  }
};

self.addEventListener("message", (event: MessageEvent<FlyGoWorkerRequest>) => {
  void handle(event.data);
});
