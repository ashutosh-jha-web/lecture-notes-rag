
"use client";

import type { Source } from "../lib/api";

type SourcePanelProps = {
  sources: Source[];
};

export default function SourcePanel({ sources }: SourcePanelProps) {
  return (
    <details className="sourcePanel">
      <summary>Sources ({sources.length})</summary>

      <div className="sourceList">
        {sources.map((source, index) => (
          <article
            key={`${source.chunk_id}-${index}`}
            className="sourceItem"
          >
            <strong>{source.source}</strong>
            <p>Page: {source.page ?? "Unknown"}</p>
            <p>Chunk: {source.chunk_id}</p>

            {source.excerpt && <blockquote>{source.excerpt}</blockquote>}
          </article>
        ))}
      </div>
    </details>
  );
}
