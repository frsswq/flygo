import { useCallback, useEffect, useMemo, useState } from "react";

import {
  DEFAULT_BOARD_SIZE,
  legalActions,
  passActionFor,
  replayMoves,
  resultOf,
  trailingPasses,
} from "@/lib/go-rules";
import type { Position } from "@/lib/go-rules";
import {
  activityOf,
  chooseAction,
  loadWebBundle,
  logitsOf,
} from "@/lib/policy";
import type {
  Dynamics,
  GraphBundle,
  PolicyBundle,
  WebBundle,
} from "@/lib/policy";

const BUNDLE_URL = `${import.meta.env.BASE_URL}flygo/`;

const messageOf = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown simulation error";

const policyMove = (
  graph: GraphBundle,
  policy: PolicyBundle,
  dynamics: Dynamics,
  position: Position
): { action: number; activity: Float32Array } => {
  const activity = activityOf(graph, policy, dynamics, position);
  return {
    action: chooseAction(logitsOf(policy, activity), legalActions(position)),
    activity,
  };
};

const readingOf = (
  bundle: WebBundle,
  policy: PolicyBundle,
  size: number
): { action: number; activity: Float32Array } =>
  policyMove(bundle.graph, policy, bundle.dynamics, replayMoves(size, []));

export const useFlyGoGame = () => {
  const [size, setSize] = useState(DEFAULT_BOARD_SIZE);
  const [moves, setMoves] = useState<number[]>([]);
  const [lastMove, setLastMove] = useState<number | null>(null);
  const [activity, setActivity] = useState<Float32Array>(
    () => new Float32Array(0)
  );
  const [loaded, setLoaded] = useState<{
    bundle: WebBundle;
    size: number;
  } | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const bundle = loaded?.size === size ? loaded.bundle : null;
  const policy = bundle?.policies.get(size);
  const isLoading = bundle === null;
  const position = useMemo(() => replayMoves(size, moves), [size, moves]);
  const legal = useMemo(() => legalActions(position), [position]);
  const result = useMemo(() => resultOf(position), [position]);
  const gameOver = trailingPasses(moves, size) >= 2;

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const next = await loadWebBundle(BUNDLE_URL, size);
        if (cancelled) {
          return;
        }
        setLoaded({ bundle: next, size });
        setActivity(
          readingOf(next, next.policies.get(size) as PolicyBundle, size)
            .activity
        );
        setFailure(null);
      } catch (error: unknown) {
        if (!cancelled) {
          setFailure(messageOf(error));
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [size]);

  const play = useCallback(
    (action: number) => {
      if (!(bundle && policy) || gameOver) {
        return;
      }
      const pass = passActionFor(size);
      const passing = action === pass;
      try {
        const afterHuman = replayMoves(size, [...moves, action]);
        const reading = policyMove(
          bundle.graph,
          policy,
          bundle.dynamics,
          afterHuman
        );
        // White passes back after a human pass, because the untrained readout
        // ranks the pass action last. The two consecutive passes then end the
        // game, and the status word reports the area score.
        const played = passing ? [action, pass] : [action, reading.action];
        setMoves([...moves, ...played]);
        setActivity(reading.activity);
        if (!passing) {
          setLastMove(reading.action === pass ? action : reading.action);
        }
      } catch (error: unknown) {
        setFailure(messageOf(error));
      }
    },
    [bundle, gameOver, moves, policy, size]
  );

  let status = "Your turn";
  if (gameOver) {
    status = result.label;
  } else if (isLoading) {
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
    isLoading,
    lastMove,
    legalActions: legal,
    play,
    reset: () => {
      setMoves([]);
      setLastMove(null);
      if (bundle && policy) {
        setActivity(readingOf(bundle, policy, size).activity);
      }
    },
    selectSize: (next: number) => {
      if (next !== size) {
        setSize(next);
        setMoves([]);
        setLastMove(null);
      }
    },
    size,
    status,
  };
};
