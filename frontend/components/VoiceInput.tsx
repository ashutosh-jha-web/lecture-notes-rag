
"use client";

import { useEffect, useRef, useState } from "react";

type SpeechRecognitionResultLike = {
  0: { transcript: string };
  isFinal: boolean;
};

type SpeechRecognitionEventLike = {
  results: ArrayLike<SpeechRecognitionResultLike>;
  resultIndex: number;
};

type SpeechRecognitionErrorEventLike = {
  error: string;
};

type SpeechRecognitionLike = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
};

type VoiceInputProps = {
  onTranscript: (text: string) => void;
  disabled?: boolean;
};

export default function VoiceInput({
  onTranscript,
  disabled = false,
}: VoiceInputProps) {
  const [listening, setListening] = useState(false);
  const [message, setMessage] = useState("");

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const activeRef = useRef(false);
  const disabledRef = useRef(disabled);

  useEffect(() => {
    disabledRef.current = disabled;

    if (disabled && recognitionRef.current) {
      activeRef.current = false;
      recognitionRef.current.abort();
      recognitionRef.current = null;
      setListening(false);
    }
  }, [disabled]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      recognitionRef.current?.abort();
      recognitionRef.current = null;
    };
  }, []);

  function startListening() {
    if (disabledRef.current || activeRef.current) return;

    const speechWindow = window as SpeechWindow;
    const Recognition =
      speechWindow.SpeechRecognition ??
      speechWindow.webkitSpeechRecognition;

    if (!Recognition) {
      setMessage(
        "Voice input is not supported in this browser. Please type your question."
      );
      return;
    }

    setMessage("");

    const recognition = new Recognition();

    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    activeRef.current = true;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      if (
        !activeRef.current ||
        disabledRef.current ||
        recognitionRef.current !== recognition
      ) {
        return;
      }

      let transcript = "";

      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }

      if (transcript.trim()) {
        onTranscript(transcript.trim());
      }
    };

    recognition.onerror = (event) => {
      if (recognitionRef.current !== recognition) return;

      activeRef.current = false;
      recognitionRef.current = null;
      setListening(false);

      if (
        event.error === "not-allowed" ||
        event.error === "service-not-allowed"
      ) {
        setMessage("Microphone permission denied. Allow microphone access.");
      } else if (event.error === "no-speech") {
        setMessage("No speech detected. Please try again.");
      } else if (event.error === "audio-capture") {
        setMessage("No microphone found. Check your microphone.");
      } else if (event.error !== "aborted") {
        setMessage(
          "Voice input failed. Please try again or type your question."
        );
      }
    };

    recognition.onend = () => {
      if (recognitionRef.current !== recognition) return;

      activeRef.current = false;
      recognitionRef.current = null;
      setListening(false);
    };

    try {
      recognition.start();
      setListening(true);
    } catch {
      activeRef.current = false;
      recognitionRef.current = null;
      setListening(false);
      setMessage("Could not start the microphone. Please try again.");
    }
  }

  function stopListening() {
    const recognition = recognitionRef.current;

    if (!recognition) return;

    // Ignore any pending speech events immediately.
    activeRef.current = false;
    recognitionRef.current = null;
    setListening(false);

    recognition.stop();
  }

  return (
    <div className="voiceInput">
      <button
        type="button"
        onClick={listening ? stopListening : startListening}
        disabled={disabled}
        aria-label={listening ? "Stop voice input" : "Start voice input"}
        aria-pressed={listening}
      >
        {listening ? "⏹ Stop listening" : "🎙 Use voice"}
      </button>

      {listening && (
        <p className="voiceStatus" role="status">
          Listening… Speak your question now.
        </p>
      )}

      {message && (
        <p className="voiceError" role="status">
          {message}
        </p>
      )}
    </div>
  );
}
