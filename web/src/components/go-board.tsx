import { RotateCcwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { PASS_ACTION } from "@/lib/board";
import type { Stone } from "@/lib/board";
import { cn } from "@/lib/utils";

interface GoBoardProps {
  board: readonly Stone[];
  disabled: boolean;
  lastComputerAction: number | null;
  legalActions: readonly number[];
  onPlay: (action: number) => void;
  onReset: () => void;
}

const stoneName = (stone: Stone): string => {
  if (stone === 1) {
    return "black";
  }
  if (stone === -1) {
    return "white";
  }
  return "empty";
};

export const GoBoard = ({
  board,
  disabled,
  lastComputerAction,
  legalActions,
  onPlay,
  onReset,
}: GoBoardProps) => {
  const legalActionSet = new Set(legalActions);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-4">
        <p className="text-muted-foreground text-sm">
          You play Black. FlyGo answers White.
        </p>
        <Button
          disabled={disabled}
          onClick={onReset}
          size="sm"
          type="button"
          variant="ghost"
        >
          <RotateCcwIcon data-icon="inline-start" />
          New game
        </Button>
      </div>

      <div className="go-board mx-auto grid aspect-square w-full max-w-96 min-w-0 grid-cols-5">
        {board.map((stone, point) => {
          const row = Math.floor(point / 5) + 1;
          const column = (point % 5) + 1;
          const pointName = stoneName(stone);
          return (
            <button
              aria-label={`Row ${row}, column ${column}: ${pointName}`}
              className={cn("go-point", pointName, {
                "last-move": lastComputerAction === point,
              })}
              disabled={disabled || stone !== 0 || !legalActionSet.has(point)}
              key={`${row}-${column}`}
              onClick={() => onPlay(point)}
              type="button"
            />
          );
        })}
      </div>

      <div className="flex items-center justify-end gap-3">
        <Button
          disabled={disabled || !legalActionSet.has(PASS_ACTION)}
          onClick={() => onPlay(PASS_ACTION)}
          type="button"
          variant="outline"
        >
          Pass
        </Button>
      </div>
    </div>
  );
};
