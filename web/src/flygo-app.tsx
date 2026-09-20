import { GoBoard } from "@/components/go-board";
import { Button } from "@/components/ui/button";
import { useFlyGoGame } from "@/hooks/use-flygo-game";
import { BOARD_SIZES, passActionFor } from "@/lib/go-rules";
import { cn } from "@/lib/utils";

const App = () => {
  const {
    activity,
    board,
    error,
    gameOver,
    isLoading,
    lastMove,
    legalActions,
    play,
    reset,
    selectSize,
    selfPlay,
    size,
    status,
    toggleSelfPlay,
  } = useFlyGoGame();

  const passAction = passActionFor(size);
  const canPass =
    !(isLoading || selfPlay) && !gameOver && legalActions.includes(passAction);

  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-4 p-4">
      <GoBoard
        activity={activity}
        board={board}
        disabled={isLoading || gameOver || selfPlay}
        lastMove={lastMove}
        legalActions={legalActions}
        onPlay={play}
        size={size}
      />

      <p
        aria-live="polite"
        className={cn(
          "text-muted-foreground w-full max-w-lg truncate text-center text-sm",
          { "text-destructive": error !== null }
        )}
      >
        {status}
      </p>

      <div className="flex w-full max-w-lg items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <select
            aria-label="Board size"
            className="border-input bg-background h-7 rounded-md border px-2 text-xs"
            onChange={(event) => selectSize(Number(event.target.value))}
            value={size}
          >
            {BOARD_SIZES.map((option) => (
              <option key={option} value={option}>
                {option}x{option}
              </option>
            ))}
          </select>

          <Button
            onClick={toggleSelfPlay}
            size="sm"
            type="button"
            variant={selfPlay ? "default" : "outline"}
          >
            {selfPlay ? "Stop" : "Self-play"}
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button onClick={reset} size="sm" type="button" variant="ghost">
            New
          </Button>
          <Button
            disabled={!canPass}
            onClick={() => play(passAction)}
            size="sm"
            type="button"
            variant="outline"
          >
            Pass
          </Button>
        </div>
      </div>
    </main>
  );
};

export default App;
