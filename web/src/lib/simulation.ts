import { z } from "zod";

import type { Stone } from "@/lib/board";

const stoneSchema = z.union([z.literal(-1), z.literal(0), z.literal(1)]);

const activeNeuronSchema = z.object({
  activity: z.number(),
  body_id: z.number().int(),
  neuron_type: z.string().nullable(),
  superclass: z.string().nullable(),
});

const simulationResponseSchema = z.object({
  activity: z.array(activeNeuronSchema),
  legal_actions: z.array(z.number().int().min(0).max(25)),
  model_status: z.string(),
  recommended_action: z.number().int().min(0).max(25),
  topology: z.string(),
});

const turnResponseSchema = z.object({
  activity: z.array(activeNeuronSchema),
  board: z.array(stoneSchema).length(25),
  computer_action: z.number().int().min(0).max(25).nullable(),
  consecutive_passes: z.number().int().min(0).max(2),
  game_over: z.boolean(),
  legal_actions: z.array(z.number().int().min(0).max(25)),
  previous_board: z.array(stoneSchema).length(25).nullable(),
});

export type ActiveNeuron = z.infer<typeof activeNeuronSchema>;
export type SimulationResponse = z.infer<typeof simulationResponseSchema>;
export type TurnResponse = z.infer<typeof turnResponseSchema>;

interface TurnRequest {
  action: number;
  board: readonly Stone[];
  consecutivePasses: number;
  previousBoard: readonly Stone[] | null;
}

const parseResponse = async <Response>(
  response: globalThis.Response,
  schema: z.ZodType<Response>
): Promise<Response> => {
  if (!response.ok) {
    throw new Error(`Simulation failed with status ${response.status}`);
  }

  const payload: unknown = await response.json();
  return schema.parse(payload);
};

export const simulatePosition = async (
  board: readonly Stone[],
  signal?: AbortSignal
): Promise<SimulationResponse> => {
  const response = await fetch("/api/simulate", {
    body: JSON.stringify({ board, to_play: 1 }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
    signal,
  });
  return parseResponse(response, simulationResponseSchema);
};

export const playTurn = async ({
  action,
  board,
  consecutivePasses,
  previousBoard,
}: TurnRequest): Promise<TurnResponse> => {
  const response = await fetch("/api/play", {
    body: JSON.stringify({
      action,
      board,
      consecutive_passes: consecutivePasses,
      previous_board: previousBoard,
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  return parseResponse(response, turnResponseSchema);
};
