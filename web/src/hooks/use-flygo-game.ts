import { useCallback, useMemo, useRef, useState } from "react";

import { useFlyGoWorker } from "@/hooks/use-flygo-worker";
import { boardViewOf, canPlay, isFinished, statusOf } from "@/lib/game-view";
import { DEFAULT_BOARD_SIZE } from "@/lib/go-rules";

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
  const [failure, setFailure] = useState<string | null>(null);
  const [searchSummary, setSearchSummary] = useState<string | null>(null);
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

  const clearMove = useCallback(
    (next: number[]) => {
      applyMoves(next);
      setLastMove(null);
      setSearchSummary(null);
      previousActionRef.current = null;
    },
    [applyMoves]
  );

  const view = useMemo(() => boardViewOf(size, moves), [size, moves]);

  const { deactivate, initialize, move, ready, readyRef, thinking } =
    useFlyGoWorker(size, SEARCH_TIME_MS, {
      acted: (result, search) => {
        const previous = previousActionRef.current;
        const nextMoves = [...movesRef.current, result.action];
        applyMoves(nextMoves);
        setActivity(result.activity);
        setLastMove(
          result.action === view.passAction ? previous : result.action
        );
        setSearchSummary(
          `${result.simulations} simulations in ${Math.round(result.elapsedMs)} ms`
        );
        previousActionRef.current = result.action;
        if (selfPlayRef.current && !isFinished(size, nextMoves)) {
          search(nextMoves);
          return;
        }
        stopSelfPlay();
      },
      failed: (message) => {
        setFailure(message);
        stopSelfPlay();
      },
      ready: (reading, search) => {
        setActivity(reading);
        if (selfPlayRef.current) {
          search(movesRef.current);
        }
      },
    });

  const play = useCallback(
    (action: number) => {
      if (
        !canPlay(
          {
            gameOver: view.gameOver,
            legal: view.legal,
            ready,
            selfPlay,
            thinking,
          },
          action
        )
      ) {
        return;
      }
      try {
        const nextMoves = [...movesRef.current, action];
        const next = boardViewOf(size, nextMoves);
        applyMoves(nextMoves);
        setFailure(null);
        setSearchSummary(null);
        previousActionRef.current = action;
        if (next.gameOver) {
          setLastMove(action === view.passAction ? lastMove : action);
          return;
        }
        move(nextMoves);
      } catch (error: unknown) {
        setFailure(messageOf(error));
      }
    },
    [
      applyMoves,
      lastMove,
      move,
      ready,
      selfPlay,
      size,
      thinking,
      view.gameOver,
      view.legal,
      view.passAction,
    ]
  );

  const toggleSelfPlay = useCallback(() => {
    if (selfPlayRef.current) {
      stopSelfPlay();
      return;
    }
    selfPlayRef.current = true;
    setSelfPlay(true);
    setFailure(null);
    clearMove([]);
    // A search request before the bundle is ready would advance the request
    // counter, and the dropped initialize response would leave the board
    // loading. The ready response starts this game instead.
    if (readyRef.current) {
      move([]);
    }
  }, [clearMove, move, readyRef, stopSelfPlay]);

  const reset = useCallback(() => {
    setFailure(null);
    clearMove([]);
    initialize();
  }, [clearMove, initialize]);

  return {
    activity: [...activity],
    board: view.board,
    error: failure,
    gameOver: view.gameOver,
    isLoading: !ready || thinking,
    lastMove,
    legalActions: view.legal,
    play,
    reset,
    selectSize: (next: number) => {
      if (next !== size) {
        setFailure(null);
        setActivity(new Float32Array(0));
        setSize(next);
        clearMove([]);
        deactivate();
      }
    },
    selfPlay,
    size,
    status: statusOf({
      failure,
      gameOver: view.gameOver,
      moveCount: moves.length,
      ready,
      result: view.result,
      searchSummary,
      selfPlay,
      thinking,
    }),
    toggleSelfPlay,
  };
};
