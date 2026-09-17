import { useEffect, useReducer } from "react";

import { DEFAULT_BOARD_SIZE, emptyBoard } from "@/lib/board";
import type { Stone } from "@/lib/board";
import { playTurn, simulatePosition } from "@/lib/simulation";
import type { ActiveNeuron, TurnResponse } from "@/lib/simulation";

interface GameState {
  activity: number[];
  board: Stone[];
  consecutivePasses: number;
  error: string | null;
  gameOver: boolean;
  isLoading: boolean;
  lastMove: number | null;
  legalActions: number[];
  previousBoard: Stone[] | null;
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
  consecutivePasses: 0,
  error: null,
  gameOver: false,
  isLoading: true,
  lastMove: null,
  legalActions: [],
  previousBoard: null,
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
        consecutivePasses: result.consecutive_passes,
        gameOver: result.game_over,
        isLoading: false,
        lastMove:
          result.computer_action === null ||
          result.computer_action === passAction
            ? action.action
            : result.computer_action,
        legalActions: result.legal_actions,
        previousBoard: result.previous_board,
      };
    }
    default: {
      return state;
    }
  }
};

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown simulation error";

export const useFlyGoGame = () => {
  const [state, dispatch] = useReducer(
    reducer,
    DEFAULT_BOARD_SIZE,
    initialState
  );

  useEffect(() => {
    const controller = new AbortController();
    const warmUp = async () => {
      try {
        const result = await simulatePosition(
          emptyBoard(DEFAULT_BOARD_SIZE),
          DEFAULT_BOARD_SIZE,
          controller.signal
        );
        if (!controller.signal.aborted) {
          dispatch({
            activity: activitiesOf(result.activity),
            legalActions: result.legal_actions,
            type: "brain-ready",
          });
        }
      } catch (error: unknown) {
        if (!controller.signal.aborted) {
          dispatch({ message: errorMessage(error), type: "failed" });
        }
      }
    };
    void warmUp();
    return () => controller.abort();
  }, []);

  const start = (size: number) => {
    dispatch({ size, type: "reset" });
    const load = async () => {
      try {
        const result = await simulatePosition(emptyBoard(size), size);
        dispatch({
          activity: activitiesOf(result.activity),
          legalActions: result.legal_actions,
          type: "brain-ready",
        });
      } catch (error: unknown) {
        dispatch({ message: errorMessage(error), type: "failed" });
      }
    };
    void load();
  };

  const play = async (action: number) => {
    if (state.isLoading || state.gameOver) {
      return;
    }
    dispatch({ type: "thinking" });
    try {
      const result = await playTurn({
        action,
        board: state.board,
        consecutivePasses: state.consecutivePasses,
        previousBoard: state.previousBoard,
        size: state.size,
      });
      dispatch({ action, result, type: "turn-complete" });
    } catch (error: unknown) {
      dispatch({ message: errorMessage(error), type: "failed" });
    }
  };

  let status = "Your turn";
  if (state.gameOver) {
    status = "Game over";
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
