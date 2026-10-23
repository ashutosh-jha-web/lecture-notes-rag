
"use client";

import { useState } from "react";
import { uploadPdfs } from "../lib/api";

export default function PdfUploader() {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [filenames, setFilenames] = useState<string[]>([]);
  const [pages, setPages] = useState(0);
  const [chunks, setChunks] = useState(0);

  async function onChange(
    e: React.ChangeEvent<HTMLInputElement>
  ) {
    const input = e.currentTarget;
    const files = Array.from(input.files ?? []);

    if (files.length === 0) return;

    const invalidFiles = files.filter(
      (file) => !file.name.toLowerCase().endsWith(".pdf")
    );

    if (invalidFiles.length > 0) {
      setStatus(
        `Only PDF files are allowed: ${invalidFiles
          .map((file) => file.name)
          .join(", ")}`
      );
      input.value = "";
      return;
    }

    setBusy(true);
    setStatus(`Uploading and indexing ${files.length} PDF file(s)...`);

    try {
      const result = await uploadPdfs(files);

      const uploadedNames = result.filenames?.length
        ? result.filenames
        : result.filename
          ? [result.filename]
          : files.map((file) => file.name);

      setFilenames(uploadedNames);
      setPages(result.pages_extracted);
      setChunks(result.chunks_created);

      setStatus(
        `Successfully uploaded ${uploadedNames.length} PDF(s).`
      );
    } catch (err) {
      setStatus(
        err instanceof Error
          ? `Upload failed: ${err.message}`
          : "Upload failed."
      );
    } finally {
      setBusy(false);
      input.value = "";
    }
  }

  return (
    <section className="panel">
      <h2>Lecture notes</h2>

      <p>
        Upload multiple PDF files together.
        A new upload batch replaces the active collection.
      </p>

      <label className="uploadButton">
        {busy ? "Processing PDFs..." : "Choose PDFs"}

        <input
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={onChange}
          disabled={busy}
        />
      </label>

      {filenames.length > 0 && (
        <div className="uploadedFiles">
          <h3>Uploaded PDFs ({filenames.length})</h3>

          <ul>
            {filenames.map((name, index) => (
              <li key={`${name}-${index}`}>📄 {name}</li>
            ))}
          </ul>

          <p>
            <strong>Total pages:</strong> {pages}
          </p>

          <p>
            <strong>Total chunks:</strong> {chunks}
          </p>
        </div>
      )}

      {status && (
        <p role="status" aria-live="polite">
          {status}
        </p>
      )}
    </section>
  );
}
