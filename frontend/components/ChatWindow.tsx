
"use client";

import { useState } from "react";
import {
  streamQuestion,
  getEvaluation,
  type AskResponse,
  type StreamEvent,
  type ConversationMessage,
  type EvaluationResult,
} from "../lib/api";
import ChatInput from "./ChatInput";
import SourcePanel from "./SourcePanel";

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: AskResponse["sources"];
  route?: string;
};

type ChatWindowProps = {
  onEvaluationChange: (evaluation: EvaluationResult | null) => void;
};

const TERMINAL_STATUSES = new Set([
  "completed",
  "partial",
  "failed",
  "error",
  "skipped",
  "success",
  "no_evaluations",
]);

const sleep = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

export default function ChatWindow({
  onEvaluationChange,
}: ChatWindowProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refreshEvaluation(
    initialEvaluation: EvaluationResult | null
  ) {
    if (initialEvaluation) {
      onEvaluationChange(initialEvaluation);

      if (TERMINAL_STATUSES.has(initialEvaluation.status)) {
        return;
      }
    }

    // Evaluation runs asynchronously, so poll for its final result.
    for (let attempt = 0; attempt < 40; attempt++) {
      await sleep(1500);

      try {
        const result = await getEvaluation();

        const evaluation: EvaluationResult = {
          ...result.latest,
          status: result.status,
        };

        onEvaluationChange(evaluation);

        if (TERMINAL_STATUSES.has(evaluation.status)) {
          return;
        }
      } catch (err) {
        console.error("Unable to refresh evaluation:", err);
      }
    }
  }

  async function handleSend(question: string) {
    if (loading) return;

    const history: ConversationMessage[] = messages
      .filter((message) => message.content.trim().length > 0)
      .slice(-10)
      .map((message) => ({
        role: message.role,
        content: message.content,
      }));

    const userMessageId = Date.now();
    const assistantMessageId = userMessageId + 1;

    setMessages((previous) => [
      ...previous,
      { id: userMessageId, role: "user", content: question },
      {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        sources: [],
      },
    ]);

    setLoading(true);
    setError("");
    onEvaluationChange(null);

    let streamCompleted = false;
    let receivedEvaluation: EvaluationResult | null = null;

    try {
      await streamQuestion(
        question,
        (event: StreamEvent) => {
          switch (event.type) {
            case "token":
              setMessages((previous) =>
                previous.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        content: message.content + event.text,
                      }
                    : message
                )
              );
              break;

            case "metadata":
              setMessages((previous) =>
                previous.map((message) =>
                  message.id === assistantMessageId
                    ? { ...message, route: event.route }
                    : message
                )
              );
              break;

            case "done":
              streamCompleted = true;

              setMessages((previous) =>
                previous.map((message) =>
                  message.id === assistantMessageId
                    ? {
                        ...message,
                        sources: event.sources ?? message.sources ?? [],
                        route: event.route ?? message.route,
                      }
                    : message
                )
              );

              receivedEvaluation = event.evaluation ?? null;

              if (receivedEvaluation) {
                onEvaluationChange(receivedEvaluation);
              }
              break;

            case "error":
              setError(event.message);
              break;
          }
        },
        history
      );

      if (!streamCompleted) {
        throw new Error("The answer stream ended before completion.");
      }

      void refreshEvaluation(receivedEvaluation);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while getting the answer."
      );
    } finally {
      setLoading(false);
    }
  }

  const latestAssistantId = [...messages]
    .reverse()
    .find((message) => message.role === "assistant")?.id;

  return (
    <section className="panel chatPanel">
      <h2>Ask your lecture notes</h2>
      <p>
        Ask questions and review the sources used to generate the answer.
      </p>

      <div className="messages" aria-live="polite">
        {messages.length === 0 && (
          <p className="emptyChat">
            Your conversation will appear here. Try asking a question
            about your uploaded PDF.
          </p>
        )}

        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            <strong>
              {message.role === "user" ? "You" : "Assistant"}
            </strong>

            <p className="messageContent">
              {message.content ||
              (message.role === "assistant" && loading)
                ? message.content || "Thinking..."
                : message.content}

              {message.role === "assistant" &&
                loading &&
                message.id === latestAssistantId &&
                message.content.length > 0 && (
                  <span className="streamingCursor" aria-hidden="true">
                    ▍
                  </span>
                )}
            </p>

            {message.role === "assistant" &&
              message.sources &&
              message.sources.length > 0 && (
                <SourcePanel sources={message.sources} />
              )}

            {message.route && (
              <small>Query route: {message.route}</small>
            )}
          </article>
        ))}

        {loading && (
          <p role="status" className="loadingMessage">
            Generating your answer...
          </p>
        )}

        {error && (
          <p role="alert" className="errorMessage">
            {error}
          </p>
        )}
      </div>

      <ChatInput onSend={handleSend} disabled={loading} />
    </section>
  );
}
