
"use client";

import { useState } from "react";
import PdfUploader from "../components/PdfUploader";
import ChatWindow from "../components/ChatWindow";
import EvaluationPanel from "../components/EvaluationPanel";
import type { EvaluationResult } from "../lib/api";

export default function Home() {
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);

  return (
    <main className="appContainer">
      <header className="appHeader">
        <h1>Lecture Notes Q&amp;A</h1>
        <p>
          Upload your lecture notes, ask questions, and explore cited sources.
        </p>
      </header>

      <PdfUploader />

      <div className="dashboardGrid">
        <ChatWindow onEvaluationChange={setEvaluation} />
        <EvaluationPanel evaluation={evaluation} />
      </div>
    </main>
  );
}
