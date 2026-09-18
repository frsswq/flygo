interface InitializeRequest {
  readonly baseUrl: string;
  readonly requestId: number;
  readonly size: number;
  readonly type: "initialize";
}

interface MoveRequest {
  readonly baseUrl: string;
  readonly moves: readonly number[];
  readonly requestId: number;
  readonly size: number;
  readonly timeMs: number;
  readonly type: "move";
}

export type FlyGoWorkerRequest = InitializeRequest | MoveRequest;

interface ReadyResponse {
  readonly activity: Float32Array;
  readonly requestId: number;
  readonly type: "ready";
}

interface MoveResponse {
  readonly action: number;
  readonly activity: Float32Array;
  readonly elapsedMs: number;
  readonly requestId: number;
  readonly simulations: number;
  readonly type: "move";
}

interface ErrorResponse {
  readonly message: string;
  readonly requestId: number;
  readonly type: "error";
}

export type FlyGoWorkerResponse = ReadyResponse | MoveResponse | ErrorResponse;
