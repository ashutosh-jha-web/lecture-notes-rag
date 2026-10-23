
"use client";

import type { EvaluationResult } from "../lib/api";

type EvaluationPanelProps = {
  evaluation: EvaluationResult | null;
};

export default function EvaluationPanel({
  evaluation,
}: EvaluationPanelProps) {
  return (
    <section className="panel evaluationPanel">
      <h2>RAGAS Evaluation</h2>

      <p>
        Quality metrics for the latest assistant response.
      </p>

      {!evaluation ? (
        <p className="evaluationEmpty">
          Ask a question to see its evaluation scores.
        </p>
      ) : (
        <>
          <p className="evaluationStatus">
            Status: <strong>{evaluation.status}</strong>
          </p>

          <div className="evaluationMetrics">
            <Metric
              label="Faithfulness"
              value={evaluation.faithfulness}
            />

            <Metric
              label="Answer Relevancy"
              value={evaluation.answer_relevancy}
            />

            <Metric
              label="Context Precision"
              value={evaluation.context_precision}
            />
          </div>
        </>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  const displayValue =
    value === null ? "Pending" : value.toFixed(4);

  return (
    <div className="evaluationMetric">
      <span className="metricLabel">{label}</span>

      <strong className="metricValue">
        {displayValue}
      </strong>

      {value !== null && (
        <p className="metricDescription">
          {(value * 100).toFixed(2)}%
        </p>
      )}
    </div>
  );
}
