"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

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

interface RouteMapProps {
  path: StationStop[];
  exploredEdges: ExploredEdge[];
}

type AnimPhase = "exploring" | "found" | "done";

const SPEED_OPTIONS: { label: string; value: number }[] = [
  { label: "1x", value: 20 },
  { label: "3x", value: 60 },
  { label: "10x", value: 200 },
];

export default function RouteMap({ path, exploredEdges }: RouteMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const canvasRenderer = useRef<L.Canvas | null>(null);

  // Layer groups
  const exploredLayerRef = useRef<L.LayerGroup | null>(null);
  const pathLayerRef = useRef<L.LayerGroup | null>(null);
  const markerLayerRef = useRef<L.LayerGroup | null>(null);

  // Animation state (refs for hot path)
  const edgeIndexRef = useRef(0);
  const speedRef = useRef(SPEED_OPTIONS[0].value);
  const isPlayingRef = useRef(true);
  const phaseRef = useRef<AnimPhase>("exploring");
  const rafIdRef = useRef<number | null>(null);
  const waveFrontRef = useRef<L.Polyline[]>([]);
  const arrivalMarkerRef = useRef<L.CircleMarker | null>(null);
  const frameCountRef = useRef(0);

  // React state for UI only
  const [animPhase, setAnimPhase] = useState<AnimPhase>("exploring");
  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [activeSpeed, setActiveSpeed] = useState(0);

  const totalEdges = exploredEdges.length;

  const pathCoords = path
    .filter((s) => s.lat !== null && s.lon !== null)
    .map((s) => [s.lat!, s.lon!] as [number, number]);

  // Draw departure & arrival markers
  const drawInitialMarkers = useCallback(
    (map: L.Map, layer: L.LayerGroup) => {
      if (path.length < 2) return;
      const dep = path[0];
      if (dep.lat && dep.lon) {
        L.circleMarker([dep.lat, dep.lon], {
          radius: 8,
          fillColor: "#22c55e",
          color: "#fff",
          weight: 2,
          fillOpacity: 1,
        })
          .bindPopup(`<b>Depart</b><br/>${dep.name}`)
          .addTo(layer);
      }
      const arr = path[path.length - 1];
      if (arr.lat && arr.lon) {
        const marker = L.circleMarker([arr.lat, arr.lon], {
          radius: 8,
          fillColor: "#ef4444",
          color: "#fff",
          weight: 2,
          fillOpacity: 0.4,
        })
          .bindPopup(`<b>Arrivee</b><br/>${arr.name}`)
          .addTo(layer);
        arrivalMarkerRef.current = marker;
      }
    },
    [path]
  );

  // Draw final path segment by segment
  const drawFinalPath = useCallback(
    (layer: L.LayerGroup, renderer: L.Canvas) => {
      if (pathCoords.length < 2) {
        phaseRef.current = "done";
        setAnimPhase("done");
        return;
      }

      let segIndex = 0;
      const drawNextSegment = () => {
        if (segIndex >= pathCoords.length - 1) {
          phaseRef.current = "done";
          setAnimPhase("done");
          return;
        }
        const from = pathCoords[segIndex];
        const to = pathCoords[segIndex + 1];

        // Glow
        L.polyline([from, to], {
          color: "#3b82f6",
          weight: 8,
          opacity: 0.3,
          renderer,
        }).addTo(layer);
        // Main line
        L.polyline([from, to], {
          color: "#3b82f6",
          weight: 3,
          opacity: 0.9,
          renderer,
        }).addTo(layer);

        segIndex++;
        setTimeout(drawNextSegment, 120);
      };

      // Also add intermediate markers
      for (let i = 1; i < path.length - 1; i++) {
        const s = path[i];
        if (s.lat && s.lon) {
          L.circleMarker([s.lat, s.lon], {
            radius: 3,
            fillColor: "#94a3b8",
            color: "#1e293b",
            weight: 1,
            fillOpacity: 0.8,
          })
            .bindPopup(s.name)
            .addTo(layer);
        }
      }

      // Light up arrival marker
      if (arrivalMarkerRef.current) {
        arrivalMarkerRef.current.setStyle({ fillOpacity: 1 });
      }

      drawNextSegment();
    },
    [path, pathCoords]
  );

  // Skip to end
  const skipToEnd = useCallback(() => {
    if (!exploredLayerRef.current || !canvasRenderer.current) return;
    const renderer = canvasRenderer.current;
    const layer = exploredLayerRef.current;

    // Cancel animation
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    // Clear existing explored edges and redraw all
    layer.clearLayers();
    waveFrontRef.current = [];

    for (const edge of exploredEdges) {
      L.polyline([edge.from, edge.to], {
        color: "#64748b",
        weight: 1,
        opacity: 0.25,
        renderer,
      }).addTo(layer);
    }

    edgeIndexRef.current = totalEdges;
    setProgress(totalEdges);
    phaseRef.current = "found";
    setAnimPhase("found");
    isPlayingRef.current = false;
    setIsPlaying(false);

    // Draw final path
    if (pathLayerRef.current) {
      pathLayerRef.current.clearLayers();
      drawFinalPath(pathLayerRef.current, renderer);
    }
  }, [exploredEdges, totalEdges, drawFinalPath]);

  // Replay animation
  const replay = useCallback(() => {
    if (!exploredLayerRef.current || !pathLayerRef.current || !markerLayerRef.current) return;

    // Cancel any running animation
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    // Clear layers
    exploredLayerRef.current.clearLayers();
    pathLayerRef.current.clearLayers();
    markerLayerRef.current.clearLayers();
    waveFrontRef.current = [];

    // Reset state
    edgeIndexRef.current = 0;
    frameCountRef.current = 0;
    phaseRef.current = "exploring";
    isPlayingRef.current = true;
    setAnimPhase("exploring");
    setProgress(0);
    setIsPlaying(true);

    // Redraw markers
    if (mapInstanceRef.current) {
      drawInitialMarkers(mapInstanceRef.current, markerLayerRef.current);
    }

    // Restart animation loop
    startAnimationLoop();
  }, [drawInitialMarkers]);

  // Animation loop
  const startAnimationLoop = useCallback(() => {
    const renderer = canvasRenderer.current;
    const layer = exploredLayerRef.current;
    if (!renderer || !layer) return;

    const animate = () => {
      if (phaseRef.current === "done" || phaseRef.current === "found") return;

      if (isPlayingRef.current && phaseRef.current === "exploring") {
        const speed = speedRef.current;
        const end = Math.min(edgeIndexRef.current + speed, totalEdges);

        // Recolor previous wave front to gray
        for (const line of waveFrontRef.current) {
          line.setStyle({ color: "#64748b", opacity: 0.25, weight: 1 });
        }
        waveFrontRef.current = [];

        // Draw new batch
        for (let i = edgeIndexRef.current; i < end; i++) {
          const edge = exploredEdges[i];
          const line = L.polyline([edge.from, edge.to], {
            color: "#22d3ee",
            weight: 1.5,
            opacity: 0.7,
            renderer,
          }).addTo(layer);
          waveFrontRef.current.push(line);
        }

        edgeIndexRef.current = end;
        frameCountRef.current++;

        // Throttled progress update (every ~10 frames)
        if (frameCountRef.current % 10 === 0 || end >= totalEdges) {
          setProgress(end);
        }

        // Check if exploration is done
        if (end >= totalEdges) {
          // Recolor last wave front
          for (const line of waveFrontRef.current) {
            line.setStyle({ color: "#64748b", opacity: 0.25, weight: 1 });
          }
          waveFrontRef.current = [];

          phaseRef.current = "found";
          setAnimPhase("found");
          setProgress(totalEdges);

          // Draw the final path
          if (pathLayerRef.current) {
            drawFinalPath(pathLayerRef.current, renderer);
          }
          return;
        }
      }

      rafIdRef.current = requestAnimationFrame(animate);
    };

    rafIdRef.current = requestAnimationFrame(animate);
  }, [exploredEdges, totalEdges, drawFinalPath]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    const next = !isPlayingRef.current;
    isPlayingRef.current = next;
    setIsPlaying(next);

    // If resuming and we were in exploring phase, restart the loop
    if (next && phaseRef.current === "exploring" && rafIdRef.current === null) {
      startAnimationLoop();
    }
  }, [startAnimationLoop]);

  // Change speed
  const changeSpeed = useCallback((index: number) => {
    speedRef.current = SPEED_OPTIONS[index].value;
    setActiveSpeed(index);
  }, []);

  // Initialize map and start animation
  useEffect(() => {
    if (!mapRef.current) return;

    // Clean up previous
    if (mapInstanceRef.current) {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }

    const renderer = L.canvas();
    canvasRenderer.current = renderer;

    const map = L.map(mapRef.current, {
      zoomControl: true,
      attributionControl: true,
      renderer,
    }).setView([46.6, 2.3], 6);
    mapInstanceRef.current = map;

    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
      }
    ).addTo(map);

    // Create layer groups
    const exploredLayer = L.layerGroup().addTo(map);
    const pathLayer = L.layerGroup().addTo(map);
    const markerLayer = L.layerGroup().addTo(map);
    exploredLayerRef.current = exploredLayer;
    pathLayerRef.current = pathLayer;
    markerLayerRef.current = markerLayer;

    // Reset animation state
    edgeIndexRef.current = 0;
    frameCountRef.current = 0;
    phaseRef.current = "exploring";
    isPlayingRef.current = true;
    waveFrontRef.current = [];
    setAnimPhase("exploring");
    setProgress(0);
    setIsPlaying(true);
    setActiveSpeed(0);
    speedRef.current = SPEED_OPTIONS[0].value;

    // Initial markers
    drawInitialMarkers(map, markerLayer);

    // Fit bounds
    if (pathCoords.length >= 2) {
      map.fitBounds(L.latLngBounds(pathCoords), { padding: [40, 40] });
    }

    // Start animation
    if (totalEdges > 0) {
      startAnimationLoop();
    } else if (pathCoords.length >= 2) {
      // No explored edges — just draw the path directly
      phaseRef.current = "found";
      setAnimPhase("found");
      drawFinalPath(pathLayer, renderer);
    }

    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      map.remove();
      mapInstanceRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, exploredEdges]);

  return (
    <div className="relative">
      <div
        ref={mapRef}
        className="w-full h-[420px] rounded-xl overflow-hidden"
      />

      {/* Animation controls overlay */}
      {totalEdges > 0 && (
        <div className="absolute bottom-3 left-3 right-3 z-[1000] flex items-center gap-2 bg-slate-900/85 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-700/50">
          {/* Play / Pause / Replay */}
          {animPhase === "done" ? (
            <button
              onClick={replay}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors"
              title="Rejouer"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
              Rejouer
            </button>
          ) : (
            <button
              onClick={togglePlay}
              className="flex items-center justify-center w-7 h-7 rounded bg-slate-700 hover:bg-slate-600 text-white transition-colors"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <rect x="6" y="4" width="4" height="16" rx="1" />
                  <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
              ) : (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>
          )}

          {/* Speed buttons */}
          {animPhase === "exploring" && (
            <div className="flex gap-0.5">
              {SPEED_OPTIONS.map((opt, i) => (
                <button
                  key={opt.label}
                  onClick={() => changeSpeed(i)}
                  className={`px-2 py-1 rounded text-xs font-mono transition-colors ${
                    activeSpeed === i
                      ? "bg-cyan-600 text-white"
                      : "bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-200"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          {/* Skip button */}
          {animPhase === "exploring" && (
            <button
              onClick={skipToEnd}
              className="flex items-center gap-1 px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs transition-colors"
              title="Passer"
            >
              Skip
              <svg
                width="10"
                height="10"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <polygon points="5,4 15,12 5,20" />
                <rect x="15" y="4" width="4" height="16" />
              </svg>
            </button>
          )}

          {/* Progress bar */}
          <div className="flex-1 flex items-center gap-2 ml-1">
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-[width] duration-150 ease-linear"
                style={{
                  width: `${totalEdges > 0 ? (progress / totalEdges) * 100 : 0}%`,
                  backgroundColor:
                    animPhase === "exploring" ? "#22d3ee" : "#3b82f6",
                }}
              />
            </div>
            <span className="text-[10px] text-slate-400 font-mono whitespace-nowrap min-w-[4rem] text-right">
              {animPhase === "done"
                ? "Termine"
                : animPhase === "found"
                  ? "Trajet..."
                  : `${progress.toLocaleString()}/${totalEdges.toLocaleString()}`}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
