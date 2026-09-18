import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  DEFAULT_BOARD_SIZE,
  legalActions,
  passActionFor,
  replayMoves,
  resultOf,
  trailingPasses,
} from "@/lib/go-rules";
import type { FlyGoWorkerResponse } from "@/lib/worker-protocol";

const BUNDLE_URL = `${import.meta.env.BASE_URL}flygo/`;
const SEARCH_TIME_MS = 1000;

const messageOf = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown simulation error";

export const useFlyGoGame = () => {
  const [size, setSize] = useState(DEFAULT_BOARD_SIZE);
  const [moves, setMoves] = useState<number[]>([]);
  const [lastMove, setLastMove] = useState<number | null>(null);
  const [activity, setActivity] = useState<Float32Array>(
    () => new Float32Array(0)
  );
  const [ready, setReady] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [searchSummary, setSearchSummary] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestRef = useRef(0);
  const pendingHumanActionRef = useRef<number | null>(null);

  const position = useMemo(() => replayMoves(size, moves), [size, moves]);
  const legal = useMemo(() => legalActions(position), [position]);
  const result = useMemo(() => resultOf(position), [position]);
  const gameOver = trailingPasses(moves, size) >= 2;

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
      if (response.type === "error") {
        setFailure(response.message);
        setThinking(false);
        return;
      }
      setActivity(response.activity);
      if (response.type === "ready") {
        setReady(true);
        return;
      }
      const pass = passActionFor(size);
      const humanAction = pendingHumanActionRef.current;
      setMoves((current) => [...current, response.action]);
      setLastMove(response.action === pass ? humanAction : response.action);
      setSearchSummary(
        `${response.simulations} simulations in ${Math.round(response.elapsedMs)} ms`
      );
      setThinking(false);
    };
    const fail = (event: ErrorEvent) => {
      setFailure(event.message || "The search worker failed");
      setThinking(false);
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
  }, [size]);

  const play = useCallback(
    (action: number) => {
      const worker = workerRef.current;
      if (
        !(worker && ready) ||
        thinking ||
        gameOver ||
        !legal.includes(action)
      ) {
        return;
      }
      try {
        const nextMoves = [...moves, action];
        replayMoves(size, nextMoves);
        setMoves(nextMoves);
        setFailure(null);
        setSearchSummary(null);
        pendingHumanActionRef.current = action;
        if (trailingPasses(nextMoves, size) >= 2) {
          setLastMove(action === passActionFor(size) ? lastMove : action);
          return;
        }
        const requestId = requestRef.current + 1;
        requestRef.current = requestId;
        setThinking(true);
        worker.postMessage(
          {
            baseUrl: BUNDLE_URL,
            moves: nextMoves,
            requestId,
            size,
            timeMs: SEARCH_TIME_MS,
            type: "move",
          },
          []
        );
      } catch (error: unknown) {
        setFailure(messageOf(error));
      }
    },
    [gameOver, lastMove, legal, moves, ready, size, thinking]
  );

  const reset = useCallback(() => {
    setMoves([]);
    setLastMove(null);
    setSearchSummary(null);
    setFailure(null);
    setThinking(false);
    const worker = workerRef.current;
    if (worker) {
      const requestId = requestRef.current + 1;
      requestRef.current = requestId;
      setReady(false);
      worker.postMessage(
        {
          baseUrl: BUNDLE_URL,
          requestId,
          size,
          type: "initialize",
        },
        []
      );
    }
  }, [size]);

  let status = searchSummary ? `Your turn · ${searchSummary}` : "Your turn";
  if (gameOver) {
    status = result.label;
  } else if (!ready || thinking) {
    status = "Thinking";
  }
  if (failure !== null) {
    status = failure;
  }

  return {
    activity: [...activity],
    board: position.board,
    error: failure,
    gameOver,
    isLoading: !ready || thinking,
    lastMove,
    legalActions: legal,
    play,
    reset,
    selectSize: (next: number) => {
      if (next !== size) {
        setReady(false);
        setThinking(false);
        setFailure(null);
        setActivity(new Float32Array(0));
        setSize(next);
        setMoves([]);
        setLastMove(null);
        setSearchSummary(null);
      }
    },
    size,
    status,
  };
};
