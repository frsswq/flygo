import { z } from "zod";

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

export type ActiveNeuron = z.infer<typeof activeNeuronSchema>;
export type SimulationResponse = z.infer<typeof simulationResponseSchema>;

export const simulatePosition = async (
  board: readonly number[],
  toPlay: 1 | -1
): Promise<SimulationResponse> => {
  const response = await fetch("/api/simulate", {
    body: JSON.stringify({ board, to_play: toPlay }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Simulation failed with status ${response.status}`);
  }

  const payload: unknown = await response.json();
  return simulationResponseSchema.parse(payload);
};
