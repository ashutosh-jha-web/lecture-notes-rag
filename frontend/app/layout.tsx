
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lecture Notes Q&A",
  description: "Ask questions about your lecture notes using RAG.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
