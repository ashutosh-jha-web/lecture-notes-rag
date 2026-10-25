
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

      <p>Quality metrics for the latest assistant response.</p>

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
              error={evaluation.metric_errors?.faithfulness}
            />

            <Metric
              label="Answer Relevancy"
              value={evaluation.answer_relevancy}
              error={evaluation.metric_errors?.answer_relevancy}
            />

            <Metric
              label="Context Precision"
              value={evaluation.context_precision}
              error={evaluation.metric_errors?.context_precision}
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
  error,
}: {
  label: string;
  value: number | null;
  error?: string;
}) {
  const displayValue =
    value !== null
      ? value.toFixed(4)
      : error
        ? "Failed"
        : "Pending";

  const readableError = error
    ? /429|rate.?limit|quota|resource.?exhausted/i.test(error)
      ? "API rate limit exceeded. Please wait about a minute and try again."
      : "Evaluation failed. Check API quota or backend logs."
    : null;

  return (
    <div className="evaluationMetric">
      <span className="metricLabel">{label}</span>

      <strong className="metricValue">{displayValue}</strong>

      {value !== null && (
        <p className="metricDescription">
          {(value * 100).toFixed(2)}%
        </p>
      )}

      {readableError && (
        <p className="metricDescription" role="status">
          {readableError}
        </p>
      )}
    </div>
  );
}
