"use client";

import { useState, useRef, useCallback } from "react";
import dynamic from "next/dynamic";

const RouteMap = dynamic(() => import("./components/RouteMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[420px] rounded-xl bg-slate-800/50 flex items-center justify-center text-slate-500 text-sm">
      Chargement de la carte...
    </div>
  ),
});

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Correction {
  original: string;
  corrected: string;
  distance: number;
}

interface StationStop {
  id: string;
  name: string;
  lat: number | null;
  lon: number | null;
}

interface ExploredEdge {
  from: [number, number];
  to: [number, number];
}

interface ResolveResult {
  transcription: string;
  corrected_text: string;
  corrections: Correction[];
  is_valid: boolean;
  departure: string | null;
  arrival: string | null;
  departure_time: string | null;
  arrival_time: string | null;
  duration_min: number | null;
  path: StationStop[];
  explored_edges: ExploredEdge[];
}

type AppState = "idle" | "recording" | "processing";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [result, setResult] = useState<ResolveResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const sendAudio = useCallback(async (blob: Blob) => {
    setState("processing");
    try {
      const formData = new FormData();
      formData.append("file", blob, "recording.webm");

      const res = await fetch(`${API_URL}/api/resolve-audio`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Erreur serveur (${res.status}): ${text}`);
      }

      const data: ResolveResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Erreur de connexion au serveur"
      );
    } finally {
      setState("idle");
    }
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);
    setResult(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      audioChunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunks.current, { type: "audio/webm" });
        sendAudio(blob);
      };

      recorder.start();
      mediaRecorder.current = recorder;
      setState("recording");
    } catch {
      setError(
        "Impossible d'acceder au microphone. Verifiez les permissions du navigateur."
      );
    }
  }, [sendAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state === "recording") {
      mediaRecorder.current.stop();
      setState("processing");
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center px-4 py-12">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-2">
          Travel Order Resolver
        </h1>
        <p className="text-slate-400 text-lg">
          Dites votre itineraire, on s&apos;occupe du reste
        </p>
      </div>

      {/* Record button */}
      <div className="relative mb-12">
        {state === "recording" && (
          <div className="absolute inset-0 rounded-full bg-red-500/30 animate-pulse-ring" />
        )}
        <button
          onClick={state === "recording" ? stopRecording : startRecording}
          disabled={state === "processing"}
          className={`
            relative z-10 w-28 h-28 rounded-full flex items-center justify-center
            text-white font-semibold text-sm transition-all duration-200
            ${
              state === "recording"
                ? "bg-red-600 hover:bg-red-700 scale-110"
                : state === "processing"
                  ? "bg-slate-600 cursor-wait"
                  : "bg-blue-600 hover:bg-blue-700 hover:scale-105"
            }
            shadow-lg shadow-blue-500/20
          `}
        >
          {state === "idle" && (
            <div className="flex flex-col items-center gap-1">
              <MicIcon />
              <span>Enregistrer</span>
            </div>
          )}
          {state === "recording" && (
            <div className="flex flex-col items-center gap-1">
              <StopIcon />
              <span>Arreter</span>
            </div>
          )}
          {state === "processing" && (
            <div className="flex flex-col items-center gap-1">
              <SpinnerIcon />
              <span>Analyse...</span>
            </div>
          )}
        </button>
      </div>

      {/* Hint */}
      {state === "idle" && !result && !error && (
        <p className="text-slate-500 text-sm mb-8 text-center max-w-md">
          Exemple : &laquo; Je veux aller de Paris a Lyon demain a 8h30 &raquo;
        </p>
      )}

      {/* Error */}
      {error && (
        <div className="w-full max-w-lg mb-8 p-4 rounded-xl bg-red-900/30 border border-red-700/50 text-red-300">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="w-full max-w-2xl space-y-4">
          {/* Transcription */}
          <Card title="Transcription brute">
            {result.transcription ? (
              <p className="text-slate-300 font-mono text-sm">
                {result.transcription}
              </p>
            ) : (
              <p className="text-yellow-400 text-sm">
                Aucun texte detecte. Parlez plus fort ou plus longtemps.
              </p>
            )}
          </Card>

          {/* Corrections */}
          {result.corrections.length > 0 && (
            <Card title="Corrections phonetiques">
              <div className="space-y-2">
                {result.corrections.map((c, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="text-red-400 line-through">
                      {c.original}
                    </span>
                    <span className="text-slate-500">&rarr;</span>
                    <span className="text-green-400 font-semibold">
                      {c.corrected}
                    </span>
                    <span className="text-slate-600 text-xs ml-auto">
                      dist: {c.distance}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Corrected text */}
          {result.corrections.length > 0 && (
            <Card title="Texte corrige">
              <p className="text-slate-300 font-mono text-sm">
                {result.corrected_text}
              </p>
            </Card>
          )}

          {/* Itinerary */}
          <Card
            title="Itineraire"
            accent={result.is_valid ? "green" : "red"}
          >
            {result.is_valid ? (
              <div className="space-y-3">
                <div className="flex items-center gap-4">
                  <StationBadge label="Depart" name={result.departure!} />
                  <div className="flex-1 border-t border-dashed border-slate-600 relative">
                    {result.duration_min && (
                      <span className="absolute -top-3 left-1/2 -translate-x-1/2 bg-slate-800 px-2 text-xs text-slate-400">
                        {result.duration_min} min
                      </span>
                    )}
                  </div>
                  <StationBadge label="Arrivee" name={result.arrival!} />
                </div>
                {(result.departure_time || result.arrival_time) && (
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>{result.departure_time || "-"}</span>
                    <span>{result.arrival_time || "-"}</span>
                  </div>
                )}

                {/* Correspondances */}
                {result.path && result.path.length > 2 && (
                  <details className="mt-2">
                    <summary className="text-xs text-slate-400 cursor-pointer hover:text-slate-300 select-none">
                      {result.path.length - 2} gare{result.path.length - 2 > 1 ? "s" : ""} intermediaire{result.path.length - 2 > 1 ? "s" : ""}
                    </summary>
                    <div className="mt-2 ml-2 border-l-2 border-blue-800/50 pl-3 space-y-1">
                      {result.path.slice(1, -1).map((stop, i) => (
                        <div key={stop.id} className="text-xs text-slate-400 flex items-center gap-2">
                          <span className="text-slate-600 w-4 text-right">{i + 1}.</span>
                          <span>{stop.name}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <p className="text-red-400">
                Itineraire invalide &mdash; impossible de resoudre la demande.
              </p>
            )}
          </Card>

          {/* Map */}
          {result.is_valid && result.path && result.path.length >= 2 && (
            <Card title="Exploration du graphe">
              <RouteMap
                path={result.path}
                exploredEdges={result.explored_edges || []}
              />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

/* ---- Sub-components ---- */

function Card({
  title,
  accent,
  children,
}: {
  title: string;
  accent?: "green" | "red";
  children: React.ReactNode;
}) {
  const borderColor =
    accent === "green"
      ? "border-green-700/50"
      : accent === "red"
        ? "border-red-700/50"
        : "border-slate-700/50";
  return (
    <div className={`rounded-xl border ${borderColor} bg-slate-800/50 p-5`}>
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
        {title}
      </h2>
      {children}
    </div>
  );
}

function StationBadge({ label, name }: { label: string; name: string }) {
  return (
    <div className="text-center">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">
        {label}
      </div>
      <div className="bg-blue-900/40 border border-blue-700/40 rounded-lg px-3 py-2 text-sm font-semibold text-blue-200">
        {name}
      </div>
    </div>
  );
}

function MicIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
      <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1" />
    </svg>
  );
}
