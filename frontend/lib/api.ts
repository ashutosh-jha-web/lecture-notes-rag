
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Source = {
  source: string;
  page: number | null;
  chunk_id: string;
  excerpt?: string;
};

export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
};

export type EvaluationMetrics = {
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
};

export type EvaluationStatus =
  | "pending"
  | "completed"
  | "partial"
  | "failed"
  | "skipped"
  | "error"
  | "success"
  | "no_evaluations"
  | string;

export type EvaluationResult = EvaluationMetrics & {
  status: EvaluationStatus;
};

export type AskResponse = {
  answer: string;
  sources: Source[];
  route?: string;
  evaluation?: EvaluationResult | null;
};

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "metadata"; route?: string }
  | {
      type: "done";
      sources?: Source[];
      route?: string;
      evaluation?: EvaluationResult | null;
    }
  | { type: "error"; message: string };

export type UploadResponse = {
  filenames?: string[];
  filename: string;
  files_uploaded?: number;
  pages_extracted: number;
  chunks_created: number;
  status: string;
};

export type DocumentStatus = {
  document_loaded: boolean;
  filename: string | null;
  filenames?: string[];
  pages_extracted: number;
  chunks_created: number;
};

export type LatestEvaluationResponse = {
  latest: EvaluationMetrics;
  status: EvaluationStatus;
};

async function readError(res: Response): Promise<string> {
  const body = await res.text();

  try {
    const parsed = JSON.parse(body);
    return parsed.detail ?? body;
  } catch {
    return body || `Request failed with status ${res.status}`;
  }
}

/**
 * Upload multiple PDFs in one request.
 * The selected batch replaces the active document collection.
 */
export async function uploadPdfs(files: File[]): Promise<UploadResponse> {
  if (files.length === 0) {
    throw new Error("Select at least one PDF to upload.");
  }

  const form = new FormData();

  for (const file of files) {
    form.append("files", file);
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await readError(res));
  }

  return res.json();
}

/** Backward-compatible single-PDF upload helper. */
export async function uploadPdf(file: File): Promise<UploadResponse> {
  return uploadPdfs([file]);
}

/**
 * Stream an answer from /ask/stream.
 * Sends recent conversation history so the backend can resolve follow-ups.
 */
export async function streamQuestion(
  question: string,
  onEvent: (event: StreamEvent) => void,
  history: ConversationMessage[] = [],
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ question, history }),
    signal,
  });

  if (!res.ok) {
    throw new Error(await readError(res));
  }

  if (!res.body) {
    throw new Error("The server did not provide a streaming response.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let receivedDone = false;

  function processEvent(rawEvent: string): boolean {
    const lines = rawEvent.split(/\r?\n/);

    const eventName =
      lines.find((line) => line.startsWith("event:"))?.slice(6).trim() ??
      "message";

    const data = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, ""))
      .join("\n");

    if (!data) {
      return false;
    }

    let payload: Record<string, unknown>;

    try {
      payload = JSON.parse(data) as Record<string, unknown>;
    } catch {
      throw new Error("Received invalid JSON in the streaming response.");
    }

    let event: StreamEvent;

    switch (eventName) {
      case "meta":
        event = {
          type: "metadata",
          route:
            typeof payload.route === "string" ? payload.route : undefined,
        };
        break;

      case "token":
        event = {
          type: "token",
          text: typeof payload.text === "string" ? payload.text : "",
        };
        break;

      case "done":
        event = {
          type: "done",
          sources: Array.isArray(payload.sources)
            ? (payload.sources as Source[])
            : [],
          route:
            typeof payload.route === "string" ? payload.route : undefined,
          evaluation:
            payload.evaluation && typeof payload.evaluation === "object"
              ? (payload.evaluation as EvaluationResult)
              : payload.evaluation === null
                ? null
                : undefined,
        };
        break;

      case "error":
        event = {
          type: "error",
          message:
            typeof payload.detail === "string"
              ? payload.detail
              : typeof payload.message === "string"
                ? payload.message
                : "The server could not finish generating the answer.",
        };
        break;

      default:
        return false;
    }

    onEvent(event);

    if (event.type === "error") {
      throw new Error(event.message);
    }

    return event.type === "done";
  }

  try {
    while (!receivedDone) {
      const { value, done } = await reader.read();

      if (done) {
        buffer += decoder.decode();
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? "";

      for (const rawEvent of events) {
        if (processEvent(rawEvent)) {
          receivedDone = true;
          break;
        }
      }
    }

    // Handle a final event without a trailing blank line.
    if (!receivedDone && buffer.trim()) {
      receivedDone = processEvent(buffer);
    }

    if (!receivedDone) {
      throw new Error("The answer stream ended before completion.");
    }
  } finally {
    reader.releaseLock();
  }
}

/** Existing non-streaming endpoint, retained for compatibility. */
export async function askQuestion(
  question: string,
  history: ConversationMessage[] = []
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, history }),
  });

  if (!res.ok) {
    throw new Error(await readError(res));
  }

  return res.json();
}

export async function getStatus(): Promise<DocumentStatus> {
  const res = await fetch(`${API_BASE}/status`, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(await readError(res));
  }

  return res.json();
}

export async function getEvaluation(): Promise<LatestEvaluationResponse> {
  const res = await fetch(`${API_BASE}/evaluation`, {
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(await readError(res));
  }

  return res.json();
}
