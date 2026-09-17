import { FlaskConicalIcon, TriangleAlertIcon } from "lucide-react";

import { ActivityMap } from "@/components/activity-map";
import { GoBoard } from "@/components/go-board";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { useFlyGoGame } from "@/hooks/use-flygo-game";
import { actionLabel } from "@/lib/board";

const App = () => {
  const {
    activity,
    board,
    error,
    gameOver,
    isLoading,
    lastComputerAction,
    legalActions,
    play,
    reset,
    status,
  } = useFlyGoGame();

  return (
    <main className="mx-auto flex min-h-svh max-w-7xl flex-col justify-center gap-8 px-5 py-8 lg:px-8">
      <section className="grid items-end gap-5 lg:grid-cols-[0.7fr_1.3fr]">
        <div>
          <p className="text-signal font-mono text-xs tracking-wide uppercase">
            Human Black · FlyGo White
          </p>
          <h1 className="mt-2 text-4xl leading-none font-semibold tracking-tighter sm:text-5xl">
            Play the fly brain.
          </h1>
        </div>
        <p className="text-muted-foreground max-w-2xl text-base leading-7">
          Place a Black stone. The frozen MaleCNS graph activates and answers
          automatically as White. Its policy is an untrained research
          demonstration.
        </p>
      </section>

      <section className="grid gap-5 lg:grid-cols-[0.85fr_1.15fr]">
        <Card className="[--card-spacing:--spacing(6)]">
          <CardHeader>
            <CardTitle>5×5 Go</CardTitle>
            <CardDescription>
              Captures, suicide, ko, and pass are checked by the server.
            </CardDescription>
            <CardAction>
              <Badge variant={gameOver ? "outline" : "secondary"}>
                {status}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent>
            <GoBoard
              board={board}
              disabled={isLoading || gameOver}
              lastComputerAction={lastComputerAction}
              legalActions={legalActions}
              onPlay={play}
              onReset={reset}
            />
          </CardContent>
        </Card>

        <Card className="[--card-spacing:--spacing(6)]">
          <CardHeader>
            <CardTitle>Connectome response</CardTitle>
            <CardDescription>
              Activity after your move passes through eight recurrent steps.
            </CardDescription>
            <CardAction>
              {isLoading ? (
                <Spinner />
              ) : (
                <Badge variant="outline">Online</Badge>
              )}
            </CardAction>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-4">
              <ActivityMap isLoading={isLoading} neurons={activity} />
              <div className="text-muted-foreground flex flex-wrap gap-5 font-mono text-xs">
                <span className="inline-flex items-center gap-1.5">
                  <i className="bg-chart-1 size-2 rounded-full" /> Positive
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <i className="bg-chart-2 size-2 rounded-full" /> Negative
                </span>
                <span>Size = |activity|</span>
              </div>
              {error ? (
                <Alert variant="destructive">
                  <TriangleAlertIcon />
                  <AlertTitle>FlyGo could not move</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              ) : null}
            </div>
          </CardContent>
          <CardFooter>
            <div>
              <span className="text-muted-foreground block font-mono text-xs tracking-wide uppercase">
                Last FlyGo move
              </span>
              <strong className="text-xl">
                {lastComputerAction === null
                  ? "Waiting for your move"
                  : actionLabel(lastComputerAction)}
              </strong>
            </div>
          </CardFooter>
        </Card>
      </section>

      <Alert>
        <FlaskConicalIcon />
        <AlertTitle>Scientific boundary</AlertTitle>
        <AlertDescription>
          The wiring comes from Janelia. The game encoding and activity dynamics
          are FlyGo experiments, not measured biological behavior.
        </AlertDescription>
      </Alert>
    </main>
  );
};

export default App;
