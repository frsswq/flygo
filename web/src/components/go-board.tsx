import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";

import { pointLabel } from "@/lib/board";
import type { Stone } from "@/lib/go-rules";

interface GoBoardProps {
  activity: readonly number[];
  board: readonly Stone[];
  disabled: boolean;
  lastMove: number | null;
  legalActions: readonly number[];
  onPlay: (action: number) => void;
  size: number;
}

interface Geometry {
  left: number;
  pad: number;
  step: number;
}

interface Theme {
  board: string;
  edge: string;
  line: string;
  negative: string;
  positive: string;
  shadow: string;
}

interface OverlayStyle extends CSSProperties {
  "--grid-columns": string;
  "--grid-inset": string;
  "--grid-span": string;
}

const BOARD_MARGIN = 0.75;
const STONE_RATIO = 0.46;
const RIM_RATIO = 0.3;
const ACTIVITY_HEIGHT = 24;
const HOSHI: Readonly<Record<number, readonly (readonly [number, number])[]>> =
  {
    19: [
      [3, 3],
      [3, 9],
      [3, 15],
      [9, 3],
      [9, 9],
      [9, 15],
      [15, 3],
      [15, 9],
      [15, 15],
    ],
  };

const geometry = (side: number, size: number): Geometry => {
  const step = side / (size - 1 + 2 * BOARD_MARGIN);
  const pad = BOARD_MARGIN * step;
  return { left: pad, pad, step };
};

const readTheme = (element: HTMLElement): Theme => {
  const style = getComputedStyle(element);
  const read = (name: string, fallback: string): string =>
    style.getPropertyValue(name).trim() || fallback;
  return {
    board: read("--board", "#d8b070"),
    edge: read("--board-edge", "#a8763c"),
    line: read("--board-line", "#33240f"),
    negative: read("--chart-2", "#4a7bd8"),
    positive: read("--chart-1", "#b8e63c"),
    shadow: read("--board-shadow", "rgb(0 0 0 / 45%)"),
  };
};

const drawStone = (
  context: CanvasRenderingContext2D,
  centerX: number,
  centerY: number,
  radius: number,
  black: boolean,
  theme: Theme,
  alpha: number
): void => {
  context.save();
  context.globalAlpha = alpha;
  context.beginPath();
  context.arc(centerX, centerY, radius, 0, Math.PI * 2);
  const gradient = context.createRadialGradient(
    centerX - radius * 0.35,
    centerY - radius * 0.4,
    radius * 0.15,
    centerX,
    centerY,
    radius
  );
  if (black) {
    gradient.addColorStop(0, "#6b6560");
    gradient.addColorStop(0.55, "#161310");
    gradient.addColorStop(1, "#000000");
  } else {
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(0.65, "#f6f2e8");
    gradient.addColorStop(1, "#cfc7b6");
  }
  context.fillStyle = gradient;
  context.shadowBlur = radius * 0.6;
  context.shadowColor = theme.shadow;
  context.shadowOffsetY = radius * 0.18;
  context.fill();
  context.restore();
};

const drawBoard = (
  canvas: HTMLCanvasElement,
  side: number,
  size: number,
  board: readonly Stone[],
  hovered: number | null,
  lastMove: number | null
): void => {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const scale = window.devicePixelRatio || 1;
  const pixelSide = Math.round(side * scale);
  if (canvas.width !== pixelSide) {
    canvas.width = pixelSide;
    canvas.height = pixelSide;
  }
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, side, side);

  const theme = readTheme(canvas);
  const { left, pad, step } = geometry(side, size);
  const radius = step * STONE_RATIO;
  const edge = pad * RIM_RATIO;
  const gridSpan = (size - 1) * step;

  context.fillStyle = theme.edge;
  context.fillStyle = theme.board;
  context.fillRect(edge, edge, side - 2 * edge, side - 2 * edge);
  context.fillRect(0, 0, side, side);

  context.strokeStyle = theme.line;
  context.lineWidth = Math.max(1, side / 640);
  context.beginPath();
  for (let index = 0; index < size; index += 1) {
    const offset = left + index * step;
    context.moveTo(left, offset);
    context.lineTo(left + gridSpan, offset);
    context.moveTo(offset, left);
    context.lineTo(offset, left + gridSpan);
  }
  context.stroke();

  context.fillStyle = theme.line;
  for (const [row, column] of HOSHI[size] ?? []) {
    context.beginPath();
    context.arc(
      left + column * step,
      left + row * step,
      step * 0.09,
      0,
      Math.PI * 2
    );
    context.fill();
  }

  for (const [point, stone] of board.entries()) {
    if (stone === 0) {
      continue;
    }
    const centerX = left + (point % size) * step;
    const centerY = left + Math.floor(point / size) * step;
    drawStone(context, centerX, centerY, radius, stone === 1, theme, 1);
  }

  if (hovered !== null && board[hovered] === 0) {
    const centerX = left + (hovered % size) * step;
    const centerY = left + Math.floor(hovered / size) * step;
    drawStone(context, centerX, centerY, radius, true, theme, 0.4);
  }

  if (
    lastMove !== null &&
    board[lastMove] !== undefined &&
    board[lastMove] !== 0
  ) {
    const centerX = left + (lastMove % size) * step;
    const centerY = left + Math.floor(lastMove / size) * step;
    context.fillStyle = board[lastMove] === 1 ? theme.positive : theme.negative;
    context.beginPath();
    context.arc(centerX, centerY, radius * 0.24, 0, Math.PI * 2);
    context.fill();
  }
};

const drawActivity = (
  canvas: HTMLCanvasElement,
  values: readonly number[],
  side: number
): void => {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const scale = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(side * scale);
  const pixelHeight = Math.round(ACTIVITY_HEIGHT * scale);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, side, ACTIVITY_HEIGHT);
  if (values.length === 0) {
    return;
  }

  const theme = readTheme(canvas);
  const peak = Math.max(...values.map((value) => Math.abs(value)), 0.0001);
  const slot = side / values.length;
  const bar = Math.max(1, slot * 0.55);
  const middle = ACTIVITY_HEIGHT / 2;

  for (const [index, value] of values.entries()) {
    const extent = (Math.abs(value) / peak) * middle * 0.92;
    const x = index * slot + (slot - bar) / 2;
    context.fillStyle = value >= 0 ? theme.positive : theme.negative;
    context.fillRect(x, value >= 0 ? middle - extent : middle, bar, extent);
  }
};

export const GoBoard = ({
  activity,
  board,
  disabled,
  lastMove,
  legalActions,
  onPlay,
  size,
}: GoBoardProps) => {
  const containerRef = useRef<HTMLFieldSetElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activityRef = useRef<HTMLCanvasElement>(null);
  const [side, setSide] = useState(0);
  const [hovered, setHovered] = useState<number | null>(null);

  useLayoutEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    const measure = () => setSide(element.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && side > 0) {
      drawBoard(canvas, side, size, board, disabled ? null : hovered, lastMove);
    }
  }, [board, disabled, hovered, lastMove, side, size]);

  useEffect(() => {
    const canvas = activityRef.current;
    if (canvas && side > 0) {
      drawActivity(canvas, activity, side);
    }
  }, [activity, side]);

  const legal = new Set(legalActions);
  const { pad, step } = geometry(side, size);
  const overlay: OverlayStyle = {
    "--grid-columns": `repeat(${size}, 1fr)`,
    "--grid-inset": `${pad - step / 2}px`,
    "--grid-span": `${step * size}px`,
  };

  return (
    <div className="flex w-full flex-col items-center gap-3">
      <fieldset
        aria-label={`Go board, ${size} by ${size}`}
        className="board-frame relative aspect-square w-full max-w-lg"
        ref={containerRef}
      >
        <canvas className="board-canvas" ref={canvasRef} />
        {side > 0 ? (
          <div className="board-grid" style={overlay}>
            {board.map((stone, point) => {
              const row = Math.floor(point / size) + 1;
              const column = (point % size) + 1;
              return (
                <button
                  aria-label={pointLabel(stone, row, column)}
                  className="board-point"
                  disabled={disabled || stone !== 0 || !legal.has(point)}
                  key={point}
                  onClick={() => onPlay(point)}
                  onPointerEnter={() => setHovered(point)}
                  onPointerLeave={() => setHovered(null)}
                  type="button"
                />
              );
            })}
          </div>
        ) : null}
      </fieldset>
      <canvas
        aria-hidden="true"
        className="board-activity max-w-lg"
        ref={activityRef}
      />
    </div>
  );
};
