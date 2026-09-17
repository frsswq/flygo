import { useEffect, useReducer } from "react";
import type { Dispatch } from "react";

import { DEFAULT_BOARD_SIZE, emptyBoard } from "@/lib/board";
import type { Stone } from "@/lib/board";
import { playTurn, simulatePosition } from "@/lib/simulation";
import type { ActiveNeuron, Score, TurnResponse } from "@/lib/simulation";

interface GameState {
  activity: number[];
  board: Stone[];
  error: string | null;
  gameOver: boolean;
  isLoading: boolean;
  lastMove: number | null;
  legalActions: number[];
  moves: number[];
  score: Score | null;
  size: number;
}

type GameAction =
  | { type: "brain-ready"; activity: number[]; legalActions: number[] }
  | { type: "failed"; message: string }
  | { type: "reset"; size: number }
  | { type: "thinking" }
  | { type: "turn-complete"; action: number; result: TurnResponse };

const activitiesOf = (neurons: ActiveNeuron[]): number[] =>
  neurons.map((neuron) => neuron.activity);

const initialState = (size: number): GameState => ({
  activity: [],
  board: emptyBoard(size),
  error: null,
  gameOver: false,
  isLoading: true,
  lastMove: null,
  legalActions: [],
  moves: [],
  score: null,
  size,
});

const reducer = (state: GameState, action: GameAction): GameState => {
  switch (action.type) {
    case "brain-ready": {
      return {
        ...state,
        activity: action.activity,
        isLoading: false,
        legalActions: action.legalActions,
      };
    }
    case "failed": {
      return { ...state, error: action.message, isLoading: false };
    }
    case "reset": {
      return initialState(action.size);
    }
    case "thinking": {
      return { ...state, error: null, isLoading: true };
    }
    case "turn-complete": {
      const { result } = action;
      const passAction = result.size * result.size;
      return {
        ...state,
        activity: activitiesOf(result.activity),
        board: result.board,
        gameOver: result.game_over,
        isLoading: false,
        lastMove:
          result.computer_action === null ||
          result.computer_action === passAction
            ? action.action
            : result.computer_action,
        legalActions: result.legal_actions,
        moves: result.moves,
        score: result.score,
      };
    }
    default: {
      return state;
    }
  }
};

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown simulation error";

const load = async (
  dispatch: Dispatch<GameAction>,
  size: number,
  signal?: AbortSignal
) => {
  try {
    const result = await simulatePosition([], size, signal);
    if (!signal?.aborted) {
      dispatch({
        activity: activitiesOf(result.activity),
        legalActions: result.legal_actions,
        type: "brain-ready",
      });
    }
  } catch (error: unknown) {
    if (!signal?.aborted) {
      dispatch({ message: errorMessage(error), type: "failed" });
    }
  }
};

export const useFlyGoGame = () => {
  const [state, dispatch] = useReducer(
    reducer,
    DEFAULT_BOARD_SIZE,
    initialState
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(dispatch, DEFAULT_BOARD_SIZE, controller.signal);
    return () => controller.abort();
  }, []);

  const start = (size: number) => {
    dispatch({ size, type: "reset" });
    void load(dispatch, size);
  };

  const play = async (action: number) => {
    if (state.isLoading || state.gameOver) {
      return;
    }
    dispatch({ type: "thinking" });
    try {
      const result = await playTurn([...state.moves, action], state.size);
      dispatch({ action, result, type: "turn-complete" });
    } catch (error: unknown) {
      dispatch({ message: errorMessage(error), type: "failed" });
    }
  };

  let status = "Your turn";
  if (state.gameOver) {
    status = state.score?.label ?? "Game over";
  } else if (state.isLoading) {
    status = "Thinking";
  }
  if (state.error !== null) {
    status = state.error;
  }

  return {
    ...state,
    play,
    reset: () => start(state.size),
    selectSize: (size: number) => {
      if (size !== state.size) {
        start(size);
      }
    },
    status,
  };
};
