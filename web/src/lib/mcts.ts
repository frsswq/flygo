import { applyMove, legalActions, resultOf } from "@/lib/go-rules";
import type { Position } from "@/lib/go-rules";
import { activityOf, logitsOf, valueOf } from "@/lib/policy";
import type { LoadedModel } from "@/lib/policy";

interface SearchNode {
  readonly consecutivePasses: number;
  readonly position: Position;
  readonly prior: number;
  readonly children: Map<number, SearchNode>;
  valueSum: number;
  visits: number;
}

export interface SearchOptions {
  readonly consecutivePasses?: number;
  readonly exploration?: number;
  readonly maxSimulations?: number;
  readonly timeMs?: number;
}

export interface SearchReading {
  readonly action: number;
  readonly activity: Float32Array;
  readonly elapsedMs: number;
  readonly rootValue: number;
  readonly simulations: number;
  readonly visits: Readonly<Record<number, number>>;
}

const nodeOf = (
  position: Position,
  consecutivePasses: number,
  prior = 1
): SearchNode => ({
  children: new Map<number, SearchNode>(),
  consecutivePasses,
  position,
  prior,
  valueSum: 0,
  visits: 0,
});

const meanValue = (node: SearchNode): number =>
  node.visits === 0 ? 0 : node.valueSum / node.visits;

const priorsOf = (
  logits: Float32Array,
  legal: readonly number[]
): Map<number, number> => {
  let peak = Number.NEGATIVE_INFINITY;
  for (const action of legal) {
    peak = Math.max(peak, logits[action]);
  }
  const priors = new Map<number, number>();
  let total = 0;
  for (const action of legal) {
    const probability = Math.exp(logits[action] - peak);
    priors.set(action, probability);
    total += probability;
  }
  for (const [action, probability] of priors) {
    priors.set(action, probability / total);
  }
  return priors;
};

const terminalValue = (node: SearchNode): number =>
  resultOf(node.position).winner * node.position.toPlay;

const expand = (
  node: SearchNode,
  model: LoadedModel
): { activity: Float32Array; value: number } => {
  if (node.consecutivePasses >= 2) {
    return {
      activity: new Float32Array(model.graph.nodeCount),
      value: terminalValue(node),
    };
  }
  const activity = activityOf(model, node.position);
  const priors = priorsOf(
    logitsOf(model.policy, activity),
    legalActions(node.position)
  );
  for (const [action, prior] of priors) {
    const passes =
      action === node.position.size * node.position.size
        ? node.consecutivePasses + 1
        : 0;
    node.children.set(
      action,
      nodeOf(applyMove(node.position, action), passes, prior)
    );
  }
  return { activity, value: valueOf(model.policy, activity) };
};

const select = (node: SearchNode, exploration: number): SearchNode => {
  const scale = Math.sqrt(Math.max(1, node.visits));
  let selected: SearchNode | undefined;
  let selectedAction = Number.POSITIVE_INFINITY;
  let selectedScore = Number.NEGATIVE_INFINITY;
  for (const [action, child] of node.children) {
    const score =
      -meanValue(child) +
      (exploration * child.prior * scale) / (1 + child.visits);
    if (
      score > selectedScore ||
      (score === selectedScore && action < selectedAction)
    ) {
      selected = child;
      selectedAction = action;
      selectedScore = score;
    }
  }
  if (!selected) {
    throw new Error("Cannot select from an unexpanded search node");
  }
  return selected;
};

const simulate = (
  root: SearchNode,
  model: LoadedModel,
  exploration: number
): Float32Array | null => {
  const path = [root];
  let node = root;
  while (node.children.size > 0 && node.consecutivePasses < 2) {
    node = select(node, exploration);
    path.push(node);
  }
  const reading = expand(node, model);
  let { value } = reading;
  for (let index = path.length - 1; index >= 0; index -= 1) {
    const visited = path[index];
    visited.visits += 1;
    visited.valueSum += value;
    value = -value;
  }
  return node === root ? reading.activity : null;
};

const hasBudget = (
  simulations: number,
  maxSimulations: number | undefined,
  deadline: number
): boolean =>
  (maxSimulations === undefined || simulations < maxSimulations) &&
  (simulations === 0 || performance.now() < deadline);

const rootChoice = (
  root: SearchNode
): { action: number; visits: Record<number, number> } => {
  let action: number | undefined;
  let bestVisits = -1;
  let bestPrior = -1;
  const visits: Record<number, number> = {};
  for (const [candidate, child] of root.children) {
    visits[candidate] = child.visits;
    const candidateIsBetter =
      child.visits > bestVisits ||
      (child.visits === bestVisits && child.prior > bestPrior) ||
      (child.visits === bestVisits &&
        child.prior === bestPrior &&
        candidate < (action ?? Infinity));
    if (candidateIsBetter) {
      action = candidate;
      bestVisits = child.visits;
      bestPrior = child.prior;
    }
  }
  if (action === undefined) {
    throw new Error("Search produced no legal action");
  }
  return { action, visits };
};

export const searchPosition = (
  model: LoadedModel,
  position: Position,
  options: SearchOptions = {}
): SearchReading => {
  const {
    consecutivePasses = 0,
    exploration = 1.5,
    maxSimulations,
    timeMs = maxSimulations === undefined ? 1000 : undefined,
  } = options;
  if (consecutivePasses >= 2) {
    throw new RangeError("Cannot search a finished game");
  }
  if (timeMs !== undefined && timeMs <= 0) {
    throw new RangeError("Search time must be positive");
  }
  if (maxSimulations !== undefined && maxSimulations < 1) {
    throw new RangeError("Simulation budget must be positive");
  }
  const root = nodeOf(position, consecutivePasses);
  const started = performance.now();
  const deadline =
    timeMs === undefined ? Number.POSITIVE_INFINITY : started + timeMs;
  let simulations = 0;
  let rootActivity: Float32Array | null = null;
  while (hasBudget(simulations, maxSimulations, deadline)) {
    const activity = simulate(root, model, exploration);
    if (activity) {
      rootActivity = activity;
    }
    simulations += 1;
  }
  if (root.children.size === 0) {
    rootActivity = expand(root, model).activity;
  }
  if (rootActivity === null) {
    throw new Error("Search produced no legal action");
  }
  const { action, visits } = rootChoice(root);
  return {
    action,
    activity: rootActivity,
    elapsedMs: performance.now() - started,
    rootValue: meanValue(root),
    simulations,
    visits,
  };
};
