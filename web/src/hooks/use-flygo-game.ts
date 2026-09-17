import { useEffect, useReducer } from "react";

import { emptyBoard } from "@/lib/board";
import type { Stone } from "@/lib/board";
import { playTurn, simulatePosition } from "@/lib/simulation";
import type { ActiveNeuron, TurnResponse } from "@/lib/simulation";

interface GameState {
  activity: ActiveNeuron[];
  board: Stone[];
  consecutivePasses: number;
  error: string | null;
  gameOver: boolean;
  isLoading: boolean;
  lastComputerAction: number | null;
  legalActions: number[];
  previousBoard: Stone[] | null;
}

type GameAction =
  | { type: "brain-ready"; activity: ActiveNeuron[]; legalActions: number[] }
  | { type: "failed"; message: string }
  | { type: "reset" }
  | { type: "thinking" }
  | { type: "turn-complete"; result: TurnResponse };

const initialState = (): GameState => ({
  activity: [],
  board: emptyBoard(),
  consecutivePasses: 0,
  error: null,
  gameOver: false,
  isLoading: true,
  lastComputerAction: null,
  legalActions: [],
  previousBoard: null,
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
      return initialState();
    }
    case "thinking": {
      return { ...state, error: null, isLoading: true };
    }
    case "turn-complete": {
      return {
        ...state,
        activity: action.result.activity,
        board: action.result.board,
        consecutivePasses: action.result.consecutive_passes,
        gameOver: action.result.game_over,
        isLoading: false,
        lastComputerAction: action.result.computer_action,
        legalActions: action.result.legal_actions,
        previousBoard: action.result.previous_board,
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
  const [state, dispatch] = useReducer(reducer, undefined, initialState);

  useEffect(() => {
    const controller = new AbortController();
    const initializeBrain = async () => {
      try {
        const result = await simulatePosition(emptyBoard(), controller.signal);
        dispatch({
          activity: result.activity,
          legalActions: result.legal_actions,
          type: "brain-ready",
        });
      } catch (error: unknown) {
        if (!controller.signal.aborted) {
          dispatch({ message: errorMessage(error), type: "failed" });
        }
      }
    };
    void initializeBrain();
    return () => controller.abort();
  }, []);

  const reset = () => {
    dispatch({ type: "reset" });
    const loadBrain = async () => {
      try {
        const result = await simulatePosition(emptyBoard());
        dispatch({
          activity: result.activity,
          legalActions: result.legal_actions,
          type: "brain-ready",
        });
      } catch (error: unknown) {
        dispatch({ message: errorMessage(error), type: "failed" });
      }
    };
    void loadBrain();
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
      });
      dispatch({ result, type: "turn-complete" });
    } catch (error: unknown) {
      dispatch({ message: errorMessage(error), type: "failed" });
    }
  };

  let status = "Your turn";
  if (state.gameOver) {
    status = "Game over";
  } else if (state.isLoading) {
    status = "FlyGo is thinking";
  }

  return { ...state, play, reset, status };
};
