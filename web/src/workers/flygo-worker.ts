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

const bundleOf = async (baseUrl: string, size: number): Promise<WebBundle> => {
  const key = `${baseUrl}:${size}`;
  const cached = bundles.get(key);
  if (cached) {
    return cached;
  }
  const loading = loadWebBundle(baseUrl, size);
  bundles.set(key, loading);
  try {
    return await loading;
  } catch (error: unknown) {
    // A transient failure must not poison the cache, but an older load must not
    // evict a newer one that already replaced it.
    if (bundles.get(key) === loading) {
      bundles.delete(key);
    }
    throw error;
  }
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
    const model = {
      dynamics: bundle.dynamics,
      graph: bundle.graph,
      policy,
    };
    if (request.type === "initialize") {
      const position = replayMoves(request.size, []);
      send({
        activity: activityOf(model, position),
        requestId: request.requestId,
        type: "ready",
      });
      return;
    }
    const position = replayMoves(request.size, request.moves);
    const reading = searchPosition(model, position, {
      consecutivePasses: trailingPasses(request.moves, request.size),
      timeMs: request.timeMs,
    });
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
