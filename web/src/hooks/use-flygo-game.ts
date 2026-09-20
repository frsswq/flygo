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
  const [selfPlay, setSelfPlay] = useState(false);
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
  const movesRef = useRef<number[]>([]);
  const selfPlayRef = useRef(false);
  const previousActionRef = useRef<number | null>(null);

  // Every search needs the complete move list, so the hook keeps the latest
  // list in a ref and mirrors it into state for rendering.
  const applyMoves = useCallback((next: number[]) => {
    movesRef.current = next;
    setMoves(next);
  }, []);

  const stopSelfPlay = useCallback(() => {
    selfPlayRef.current = false;
    setSelfPlay(false);
  }, []);

  const requestMove = useCallback(
    (nextMoves: readonly number[]) => {
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
          moves: nextMoves,
          requestId,
          size,
          timeMs: SEARCH_TIME_MS,
          type: "move",
        },
        []
      );
    },
    [size]
  );

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
        stopSelfPlay();
        return;
      }
      setActivity(response.activity);
      if (response.type === "ready") {
        setReady(true);
        if (selfPlayRef.current) {
          requestMove(movesRef.current);
        }
        return;
      }
      const pass = passActionFor(size);
      const previous = previousActionRef.current;
      const nextMoves = [...movesRef.current, response.action];
      applyMoves(nextMoves);
      setLastMove(response.action === pass ? previous : response.action);
      setSearchSummary(
        `${response.simulations} simulations in ${Math.round(response.elapsedMs)} ms`
      );
      previousActionRef.current = response.action;
      if (selfPlayRef.current && trailingPasses(nextMoves, size) < 2) {
        requestMove(nextMoves);
        return;
      }
      stopSelfPlay();
      setThinking(false);
    };
    const fail = (event: ErrorEvent) => {
      setFailure(event.message || "The search worker failed");
      setThinking(false);
      stopSelfPlay();
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
  }, [requestMove, applyMoves, size, stopSelfPlay]);

  const play = useCallback(
    (action: number) => {
      const worker = workerRef.current;
      if (
        selfPlay ||
        !(worker && ready) ||
        thinking ||
        gameOver ||
        !legal.includes(action)
      ) {
        return;
      }
      try {
        const nextMoves = [...movesRef.current, action];
        replayMoves(size, nextMoves);
        applyMoves(nextMoves);
        setFailure(null);
        setSearchSummary(null);
        previousActionRef.current = action;
        if (trailingPasses(nextMoves, size) >= 2) {
          setLastMove(action === passActionFor(size) ? lastMove : action);
          return;
        }
        requestMove(nextMoves);
      } catch (error: unknown) {
        setFailure(messageOf(error));
      }
    },
    [
      gameOver,
      lastMove,
      legal,
      ready,
      requestMove,
      selfPlay,
      applyMoves,
      size,
      thinking,
    ]
  );

  const startSelfPlay = useCallback(() => {
    selfPlayRef.current = true;
    setSelfPlay(true);
    applyMoves([]);
    setLastMove(null);
    setSearchSummary(null);
    setFailure(null);
    previousActionRef.current = null;
    requestMove([]);
  }, [requestMove, applyMoves]);

  const toggleSelfPlay = useCallback(() => {
    if (selfPlayRef.current) {
      stopSelfPlay();
      return;
    }
    startSelfPlay();
  }, [startSelfPlay, stopSelfPlay]);

  const reset = useCallback(() => {
    applyMoves([]);
    setLastMove(null);
    setSearchSummary(null);
    setFailure(null);
    setThinking(false);
    previousActionRef.current = null;
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
  }, [applyMoves, size]);

  let status = "Your turn";
  if (selfPlay) {
    status = searchSummary
      ? `FlyGo vs FlyGo · move ${moves.length} · ${searchSummary}`
      : `FlyGo vs FlyGo · move ${moves.length + 1}`;
  } else if (searchSummary) {
    status = `Your turn · ${searchSummary}`;
  }
  if (gameOver) {
    status = result.label;
  } else if (!selfPlay && failure === null && (!ready || thinking)) {
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
        applyMoves([]);
        setLastMove(null);
        setSearchSummary(null);
        previousActionRef.current = null;
      }
    },
    selfPlay,
    size,
    status,
    toggleSelfPlay,
  };
};
