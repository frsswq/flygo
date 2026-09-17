/**
 * Browser inference over the exported MaleCNS graph and policy weights.
 *
 * The binaries are little-endian, as written by `flygo export-web`. See
 * `src/flygo/export.py` for the exact byte layout and the dynamics constants.
 * The recurrent wiring stays frozen: only the encoder and readout are weights.
 */

import { z } from "zod";

import type { Position } from "@/lib/go-rules";
import { pointsFor } from "@/lib/go-rules";

export const GRAPH_MAGIC = 0x47_59_4c_46;
export const POLICY_MAGIC = 0x50_59_4c_46;
export const BUNDLE_VERSION = 1;
export const DEFAULT_STEPS = 8;

const manifestSchema = z.object({
  dynamics: z.object({
    recurrent_gain: z.number(),
    retention: z.number(),
    steps: z.number().int().min(1).max(32),
  }),
  graph: z.object({
    edge_count: z.number().int().min(1),
    file: z.string(),
    node_count: z.number().int().min(1),
    sha256: z.string(),
  }),
  policies: z.array(
    z.object({
      action_count: z.number().int().min(2),
      feature_count: z.number().int().min(1),
      file: z.string(),
      seed: z.number().int(),
      size: z.number().int().min(2),
    })
  ),
  ruleset: z.string(),
});

export type Manifest = z.infer<typeof manifestSchema>;
export type PolicyEntry = Manifest["policies"][number];

export interface GraphBundle {
  readonly edgeCount: number;
  readonly incomingStrength: Float32Array;
  readonly nodeCount: number;
  readonly nodeIds: Int32Array;
  readonly sources: Uint32Array;
  readonly targets: Uint32Array;
  readonly weights: Float32Array;
}

export interface PolicyBundle {
  readonly actionCount: number;
  readonly encoder: Float32Array;
  readonly featureCount: number;
  readonly readout: Float32Array;
  readonly size: number;
}

export interface Dynamics {
  readonly recurrentGain: number;
  readonly retention: number;
  readonly steps: number;
}

export interface WebBundle {
  readonly dynamics: Dynamics;
  readonly graph: GraphBundle;
  readonly policies: Map<number, PolicyBundle>;
  readonly ruleset: string;
}

const readHeader = (buffer: ArrayBuffer, length: number): DataView => {
  if (buffer.byteLength < length) {
    throw new RangeError(
      `Bundle needs at least ${length} header bytes, got ${buffer.byteLength}`
    );
  }
  return new DataView(buffer);
};

export const parseGraph = (buffer: ArrayBuffer): GraphBundle => {
  const view = readHeader(buffer, 16);
  const magic = view.getUint32(0, true);
  const version = view.getUint32(4, true);
  if (magic !== GRAPH_MAGIC) {
    throw new RangeError("Graph bundle has an unknown magic number");
  }
  if (version !== BUNDLE_VERSION) {
    throw new RangeError(`Graph bundle version ${version} is not supported`);
  }
  const nodeCount = view.getUint32(8, true);
  const edgeCount = view.getUint32(12, true);
  const expected = 16 + nodeCount * 4 + edgeCount * 4 * 3 + nodeCount * 4;
  if (buffer.byteLength !== expected) {
    throw new RangeError(
      `Graph bundle holds ${buffer.byteLength} bytes, expected ${expected}`
    );
  }
  let offset = 16;
  const nodeIds = new Int32Array(buffer, offset, nodeCount);
  offset += nodeCount * 4;
  const sources = new Uint32Array(buffer, offset, edgeCount);
  offset += edgeCount * 4;
  const targets = new Uint32Array(buffer, offset, edgeCount);
  offset += edgeCount * 4;
  const weights = new Float32Array(buffer, offset, edgeCount);
  offset += edgeCount * 4;
  const incomingStrength = new Float32Array(buffer, offset, nodeCount);
  return {
    edgeCount,
    incomingStrength,
    nodeCount,
    nodeIds,
    sources,
    targets,
    weights,
  };
};

export const parsePolicy = (buffer: ArrayBuffer): PolicyBundle => {
  const view = readHeader(buffer, 24);
  const magic = view.getUint32(0, true);
  const version = view.getUint32(4, true);
  if (magic !== POLICY_MAGIC) {
    throw new RangeError("Policy bundle has an unknown magic number");
  }
  if (version !== BUNDLE_VERSION) {
    throw new RangeError(`Policy bundle version ${version} is not supported`);
  }
  const size = view.getUint32(8, true);
  const nodeCount = view.getUint32(12, true);
  const featureCount = view.getUint32(16, true);
  const actionCount = view.getUint32(20, true);
  const expected =
    24 + nodeCount * featureCount * 4 + actionCount * nodeCount * 4;
  if (buffer.byteLength !== expected) {
    throw new RangeError(
      `Policy bundle holds ${buffer.byteLength} bytes, expected ${expected}`
    );
  }
  const encoder = new Float32Array(buffer, 24, nodeCount * featureCount);
  const readout = new Float32Array(
    buffer,
    24 + nodeCount * featureCount * 4,
    actionCount * nodeCount
  );
  return { actionCount, encoder, featureCount, readout, size };
};

export const parseManifest = (payload: unknown): Manifest =>
  manifestSchema.parse(payload);

export const dynamicsOf = (manifest: Manifest): Dynamics => ({
  recurrentGain: manifest.dynamics.recurrent_gain,
  retention: manifest.dynamics.retention,
  steps: manifest.dynamics.steps,
});

export const featuresOf = (position: Position): Float32Array => {
  const { board, size, toPlay } = position;
  const points = pointsFor(size);
  const features = new Float32Array(2 * points + 1);
  for (let index = 0; index < points; index += 1) {
    const stone = board[index];
    if (stone === toPlay) {
      features[index] = 1;
    } else if (stone === -toPlay) {
      features[points + index] = 1;
    }
  }
  features[2 * points] = toPlay;
  return features;
};

const encoderInput = (
  graph: GraphBundle,
  policy: PolicyBundle,
  features: Float32Array
): Float32Array => {
  const input = new Float32Array(graph.nodeCount);
  for (let node = 0; node < graph.nodeCount; node += 1) {
    const offset = node * policy.featureCount;
    let sum = 0;
    for (let feature = 0; feature < policy.featureCount; feature += 1) {
      sum += policy.encoder[offset + feature] * features[feature];
    }
    input[node] = Math.fround(Math.tanh(sum));
  }
  return input;
};

const stepGraph = (
  graph: GraphBundle,
  dynamics: Dynamics,
  state: Float32Array,
  externalInput: Float32Array
): Float32Array => {
  const retention = Math.fround(dynamics.retention);
  const gain = Math.fround(dynamics.recurrentGain);
  const drive = new Float64Array(graph.nodeCount);
  for (let edge = 0; edge < graph.edgeCount; edge += 1) {
    drive[graph.targets[edge]] +=
      graph.weights[edge] * state[graph.sources[edge]];
  }
  const next = new Float32Array(graph.nodeCount);
  for (let node = 0; node < graph.nodeCount; node += 1) {
    const recurrent = Math.fround(
      Math.fround(drive[node]) / graph.incomingStrength[node]
    );
    const total = Math.fround(
      Math.fround(
        Math.fround(retention * state[node]) + Math.fround(gain * recurrent)
      ) + externalInput[node]
    );
    next[node] = Math.fround(Math.tanh(total));
  }
  return next;
};

export const runGraph = (
  graph: GraphBundle,
  dynamics: Dynamics,
  externalInput: Float32Array,
  steps: number = dynamics.steps
): Float32Array => {
  if (steps < 1) {
    throw new RangeError("Simulation steps must be positive");
  }
  let state: Float32Array = new Float32Array(graph.nodeCount);
  for (let step = 0; step < steps; step += 1) {
    state = stepGraph(graph, dynamics, state, externalInput);
  }
  return state;
};

export const activityOf = (
  graph: GraphBundle,
  policy: PolicyBundle,
  dynamics: Dynamics,
  position: Position,
  steps: number = dynamics.steps
): Float32Array =>
  runGraph(
    graph,
    dynamics,
    encoderInput(graph, policy, featuresOf(position)),
    steps
  );

export const logitsOf = (
  policy: PolicyBundle,
  activity: Float32Array
): Float32Array => {
  const logits = new Float32Array(policy.actionCount);
  for (let action = 0; action < policy.actionCount; action += 1) {
    const offset = action * activity.length;
    let sum = 0;
    for (let node = 0; node < activity.length; node += 1) {
      sum += policy.readout[offset + node] * activity[node];
    }
    logits[action] = Math.fround(sum);
  }
  return logits;
};

/** Pick the strongest legal action, keeping the first of any tie. */
export const chooseAction = (
  logits: Float32Array,
  legalActions: readonly number[]
): number => {
  if (legalActions.length === 0) {
    throw new RangeError("A position must have at least one legal action");
  }
  let [best] = legalActions;
  for (const action of legalActions) {
    if (logits[action] > logits[best]) {
      best = action;
    }
  }
  return best;
};

export const strongestNeurons = (
  activity: Float32Array,
  limit = 40
): number[] => {
  const indices = Array.from({ length: activity.length }, (_, index) => index);
  return indices
    .toSorted((a, b) => Math.abs(activity[b]) - Math.abs(activity[a]))
    .slice(0, limit);
};

export const strongestActivity = (
  activity: Float32Array,
  limit = 40
): number[] =>
  strongestNeurons(activity, limit).map((index) => activity[index]);

/** Fetch a bundle written by `flygo export-web`. */
export const loadWebBundle = async (
  baseUrl: string,
  size: number
): Promise<WebBundle> => {
  const root = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const manifestResponse = await fetch(`${root}manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error(
      `Bundle manifest failed with status ${manifestResponse.status}`
    );
  }
  const manifest = parseManifest(await manifestResponse.json());
  const entry = manifest.policies.find((policy) => policy.size === size);
  if (!entry) {
    throw new RangeError(
      `The bundle has no policy for a ${size}x${size} board`
    );
  }
  const [graphResponse, policyResponse] = await Promise.all([
    fetch(`${root}${manifest.graph.file}`),
    fetch(`${root}${entry.file}`),
  ]);
  if (!(graphResponse.ok && policyResponse.ok)) {
    throw new Error("Bundle files failed to load");
  }
  return {
    dynamics: dynamicsOf(manifest),
    graph: parseGraph(await graphResponse.arrayBuffer()),
    policies: new Map([
      [size, parsePolicy(await policyResponse.arrayBuffer())],
    ]),
    ruleset: manifest.ruleset,
  };
};
