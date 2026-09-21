import { useCallback, useEffect, useRef, useState } from "react";

import type { FlyGoWorkerResponse } from "@/lib/worker-protocol";

const BUNDLE_URL = `${import.meta.env.BASE_URL}flygo/`;

/** Ask the worker to search the position that ``moves`` reaches. */
type SearchRequest = (moves: readonly number[]) => void;

interface ActionResult {
  readonly action: number;
  readonly activity: Float32Array;
  readonly elapsedMs: number;
  readonly simulations: number;
}

export interface FlyGoWorkerHandlers {
  /** The worker answered a search request. */
  readonly acted: (result: ActionResult, search: SearchRequest) => void;
  /** The worker rejected a request or the worker itself failed. */
  readonly failed: (message: string) => void;
  /** The bundle is loaded for the current board size. */
  readonly ready: (activity: Float32Array, search: SearchRequest) => void;
}

export interface FlyGoWorker {
  /**
   * Forget the current session state without touching the worker, for a board
   * size change that replaces the worker anyway.
   */
  readonly deactivate: () => void;
  /** Replace the current search session with a fresh initialization. */
  readonly initialize: () => void;
  /** Search the position that ``moves`` reaches. */
  readonly move: SearchRequest;
  readonly ready: boolean;
  readonly readyRef: React.RefObject<boolean>;
  readonly thinking: boolean;
}

/**
 * Own one search worker and the request generation that guards its responses.
 *
 * A response is dropped when a newer request has already been sent, so a stale
 * search cannot commit to a game that started after it. The worker is recreated
 * for each board size; the caller deactivates the session as it changes size.
 */
export const useFlyGoWorker = (
  size: number,
  searchTimeMs: number,
  handlers: FlyGoWorkerHandlers
): FlyGoWorker => {
  const [ready, setReady] = useState(false);
  const [thinking, setThinking] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const requestRef = useRef(0);
  // Readiness is also kept in a ref, because callers must know whether the
  // bundle has loaded without waiting for a render.
  const readyRef = useRef(false);
  // Handlers are read when a response arrives, so a re-render never recreates
  // the worker. This effect is declared before the worker effect so the
  // handlers are current before any response can arrive.
  const handlersRef = useRef(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  });

  const markReady = useCallback((next: boolean) => {
    readyRef.current = next;
    setReady(next);
  }, []);

  const move = useCallback<SearchRequest>(
    (moves) => {
      const worker = workerRef.current;
      if (!worker) {
        return;
      }
      const requestId = requestRef.current + 1;
      requestRef.current = requestId;
      setThinking(true);
      worker.postMessage(
        {
          baseUrl: BUNDLE_URL,
          moves,
          requestId,
          size,
          timeMs: searchTimeMs,
          type: "move",
        },
        []
      );
    },
    [searchTimeMs, size]
  );

  const deactivate = useCallback(() => {
    markReady(false);
    setThinking(false);
  }, [markReady]);

  const initialize = useCallback(() => {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    deactivate();
    worker.postMessage(
      {
        baseUrl: BUNDLE_URL,
        requestId,
        size,
        type: "initialize",
      },
      []
    );
  }, [deactivate, size]);

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/flygo-worker.ts", import.meta.url),
      { type: "module" }
    );
    workerRef.current = worker;
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;

    const receive = (event: MessageEvent<FlyGoWorkerResponse>) => {
      const response = event.data;
      if (response.requestId !== requestRef.current) {
        return;
      }
      const { current: events } = handlersRef;
      setThinking(false);
      if (response.type === "error") {
        events.failed(response.message);
        return;
      }
      if (response.type === "ready") {
        markReady(true);
        events.ready(response.activity, move);
        return;
      }
      events.acted(
        {
          action: response.action,
          activity: response.activity,
          elapsedMs: response.elapsedMs,
          simulations: response.simulations,
        },
        move
      );
    };
    const fail = (event: ErrorEvent) => {
      setThinking(false);
      handlersRef.current.failed(event.message || "The search worker failed");
    };
    worker.addEventListener("message", receive);
    worker.addEventListener("error", fail);
    worker.postMessage(
      {
        baseUrl: BUNDLE_URL,
        requestId,
        size,
        type: "initialize",
      },
      []
    );
    return () => {
      worker.removeEventListener("message", receive);
      worker.removeEventListener("error", fail);
      worker.terminate();
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
    };
  }, [markReady, move, size]);

  return { deactivate, initialize, move, ready, readyRef, thinking };
};
