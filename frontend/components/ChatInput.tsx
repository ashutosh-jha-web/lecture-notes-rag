
"use client";

import { useState } from "react";
import VoiceInput from "./VoiceInput";

type ChatInputProps = {
  onSend: (question: string) => void;
  disabled?: boolean;
};

export default function ChatInput({
  onSend,
  disabled = false,
}: ChatInputProps) {
  const [question, setQuestion] = useState("");

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const trimmed = question.trim();

    if (!trimmed || disabled) return;

    // Clear the input immediately after submitting.
    setQuestion("");

    try {
      onSend(trimmed);
    } catch (error) {
      // Restore the question if submission throws synchronously.
      setQuestion(trimmed);
      console.error("Failed to submit question:", error);
    }
  }

  function handleKeyDown(
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      if (!disabled && question.trim()) {
        e.currentTarget.form?.requestSubmit();
      }
    }
  }

  return (
    <form className="chatInput" onSubmit={handleSubmit}>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your lecture notes..."
        rows={2}
        disabled={disabled}
        aria-label="Your question"
      />

      <div className="chatInputActions">
        <VoiceInput
          onTranscript={setQuestion}
          disabled={disabled}
        />

        <button
          type="submit"
          disabled={disabled || !question.trim()}
        >
          {disabled ? "Thinking..." : "Ask"}
        </button>
      </div>
    </form>
  );
}
