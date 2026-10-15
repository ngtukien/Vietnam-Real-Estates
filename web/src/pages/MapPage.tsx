import "maplibre-gl/dist/maplibre-gl.css";
import type * as GeoJSON from "geojson";
import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import type { ExpressionSpecification, GeoJSONSource, StyleSpecification } from "maplibre-gl";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { errorMessage } from "@/api/client";
import { useFilterOptions, useGeoAreas } from "@/api/queries";
import type { GeoItem, GeoLevel } from "@/api/schema";
import { Alert, Card, CardContent, DataTable, Label, NativeSelect, Spinner } from "@/components/ui/primitives";
import { formatInt, formatPercent, formatPriceM2 } from "@/lib/format";

type Feature = GeoJSON.Feature<GeoJSON.Geometry, { key: string | null; name: string; province?: string | null }>;
type Collection = GeoJSON.FeatureCollection<GeoJSON.Geometry, Feature["properties"]>;

// Worker của MapLibre là file riêng: chỉ rõ URL để Vite đóng gói/phục vụ nó (dev lẫn build).
maplibregl.setWorkerUrl(workerUrl);

const geoCache = new Map<string, Promise<Collection>>();
function loadGeo(name: "provinces" | "districts"): Promise<Collection> {
  let p = geoCache.get(name);
  if (!p) {
    p = fetch(`/geo/${name}.json`).then((r) => {
      if (!r.ok) throw new Error(`Không tải được ranh giới ${name} (HTTP ${r.status})`);
      return r.json() as Promise<Collection>;
    });
    geoCache.set(name, p);
  }
  return p;
}

// Thang màu giá/m² (triệu đồng) — tuần tự, đọc được ở cả nền sáng/tối.
const STOPS: [number, string][] = [
  [10, "#e0f2f1"],
  [25, "#80cbc4"],
  [50, "#26a69a"],
  [80, "#00796b"],
  [130, "#004d40"],
];

const FILL_COLOR: ExpressionSpecification = [
  "case",
  ["==", ["typeof", ["get", "median"]], "number"],
  ["interpolate", ["linear"], ["get", "median"], ...STOPS.flat()],
  "#cbd5e1",
];

/** Style không có tile ngoài: chỉ nền + lớp GeoJSON cục bộ (yêu cầu phase C3). */
function baseStyle(dark: boolean): StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [{ id: "background", type: "background", paint: { "background-color": dark ? "#0f172a" : "#f1f5f9" } }],
  };
}

function bbox(features: Feature[]): [[number, number], [number, number]] | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const visit = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number" && typeof c[1] === "number") {
      minX = Math.min(minX, c[0]);
      maxX = Math.max(maxX, c[0]);
      minY = Math.min(minY, c[1]);
      maxY = Math.max(maxY, c[1]);
    } else if (Array.isArray(c)) c.forEach(visit);
  };
  for (const f of features) if (f.geometry.type !== "GeometryCollection") visit(f.geometry.coordinates);
  return Number.isFinite(minX) ? [[minX, minY], [maxX, maxY]] : null;
}

function joinItems(collection: Collection, items: GeoItem[], province: string | null): Collection {
  const byKey = new Map(items.map((i) => [i.key, i]));
  const features = collection.features
    .filter((f) => province === null || f.properties.province === province)
    .map((f) => {
      const item = f.properties.key ? byKey.get(f.properties.key) : undefined;
      return {
        ...f,
        properties: { ...f.properties, n: item?.n_listings ?? 0, median: item?.median_price_m2_mil ?? null },
      };
    });
  return { type: "FeatureCollection", features };
}

function pointsOf(items: GeoItem[]): GeoJSON.FeatureCollection {
  const maxN = Math.max(1, ...items.map((i) => i.n_listings));
  return {
    type: "FeatureCollection",
    features: items
      .filter((i) => i.lat !== null && i.lon !== null)
      .map((i) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [i.lon as number, i.lat as number] },
        properties: { name: i.name, n: i.n_listings, r: 3 + 17 * Math.sqrt(i.n_listings / maxN), median: i.median_price_m2_mil },
      })),
  };
}

export default function MapPage() {
  const id = useId();
  const [level, setLevel] = useState<GeoLevel>("province");
  const [province, setProvince] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const { data: options } = useFilterOptions();
  const { data, isLoading, error } = useGeoAreas(level, province);
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);

  useEffect(() => {
    if (!container.current) return;
    const dark = document.documentElement.classList.contains("dark");
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: container.current,
        style: baseStyle(dark),
        center: [106, 16],
        zoom: 4.3,
        attributionControl: false,
      });
    } catch (e) {
      const message = `Không khởi tạo được bản đồ (WebGL): ${e instanceof Error ? e.message : String(e)}`;
      queueMicrotask(() => setGeoError(message));
      return;
    }
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: "geoBoundaries (CC BY 3.0 IGO / PD)" }));
    map.on("load", () => {
      map.addSource("areas", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addSource("points", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: "areas-fill", type: "fill", source: "areas", paint: { "fill-color": FILL_COLOR, "fill-opacity": 0.85 } });
      map.addLayer({
        id: "areas-line",
        type: "line",
        source: "areas",
        paint: { "line-color": dark ? "#334155" : "#ffffff", "line-width": 0.6 },
      });
      map.addLayer({
        id: "points",
        type: "circle",
        source: "points",
        paint: {
          "circle-radius": ["get", "r"],
          "circle-color": "#c2410c",
          "circle-opacity": 0.55,
          "circle-stroke-color": "#7c2d12",
          "circle-stroke-width": 1,
        },
      });
      map.on("click", "areas-fill", (e) => {
        const f = e.features?.[0];
        if (f) setSelected(String(f.properties?.key ?? f.properties?.name ?? ""));
      });
      map.on("mouseenter", "areas-fill", () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", "areas-fill", () => (map.getCanvas().style.cursor = ""));
      setReady(true);
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !data) return;
    let cancelled = false;
    loadGeo(level === "province" ? "provinces" : "districts")
      .then((geo) => {
        if (cancelled) return;
        const joined = joinItems(geo, data.items, level === "province" ? null : province);
        (map.getSource("areas") as GeoJSONSource).setData(joined);
        (map.getSource("points") as GeoJSONSource).setData(pointsOf(data.items));
        const box = bbox(joined.features);
        if (box) map.fitBounds(box, { padding: 24, duration: 0 });
        setGeoError(null);
      })
      .catch((e: unknown) => setGeoError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [ready, data, level, province]);

  const sorted = useMemo(() => [...(data?.items ?? [])].sort((a, b) => b.n_listings - a.n_listings), [data]);
  const selectedItem = sorted.find((i) => i.key === selected || i.name === selected);

  return (
    <div className="grid gap-4">
      <h1 className="text-lg font-semibold">Bản đồ giá/m² theo khu vực</h1>
      <div className="flex flex-wrap items-end gap-4">
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-level`}>Cấp</Label>
          <NativeSelect
            id={`${id}-level`}
            value={level}
            className="w-44"
            onChange={(e) => {
              setLevel(e.target.value as GeoLevel);
              setSelected(null);
            }}
          >
            <option value="province">Tỉnh/thành</option>
            <option value="district">Quận/huyện</option>
          </NativeSelect>
        </div>
        {level === "district" ? (
          <div className="grid gap-1.5">
            <Label htmlFor={`${id}-prov`}>Tỉnh/thành</Label>
            <NativeSelect
              id={`${id}-prov`}
              className="w-56"
              value={province ?? ""}
              onChange={(e) => setProvince(e.target.value || null)}
            >
              <option value="">(chọn tỉnh)</option>
              {options?.provinces.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </NativeSelect>
          </div>
        ) : null}
        {data ? (
          <p className="text-sm text-muted-foreground" data-testid="coverage">
            Độ phủ toạ độ: {formatPercent(data.coverage)} · {data.items.length} khu vực
          </p>
        ) : null}
        {isLoading ? <Spinner /> : null}
      </div>
      {level === "district" && !province ? <Alert>Chọn tỉnh để xem các quận/huyện.</Alert> : null}
      {error ? <Alert tone="error">{errorMessage(error)}</Alert> : null}
      {geoError ? <Alert tone="error">{geoError}</Alert> : null}

      <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
        <div
          ref={container}
          role="region"
          aria-label="Bản đồ: màu = giá/m² trung vị, chấm tròn = số tin. Dữ liệu dạng bảng ở dưới."
          className="h-[520px] overflow-hidden rounded-lg border"
          data-testid="map"
        />
        <Card>
          <CardContent className="grid gap-3 pt-4 text-sm">
            <h2 className="font-medium">Chú giải</h2>
            <ul className="grid gap-1" aria-label="Thang màu giá/m²">
              {STOPS.map(([v, c]) => (
                <li key={v} className="flex items-center gap-2">
                  <span className="inline-block size-4 rounded-sm border" style={{ background: c }} aria-hidden />≥{" "}
                  {formatPriceM2(v)}
                </li>
              ))}
              <li className="flex items-center gap-2">
                <span className="inline-block size-4 rounded-sm border bg-slate-300" aria-hidden /> không có dữ liệu
              </li>
            </ul>
            <p className="text-xs text-muted-foreground">Chấm tròn cam: diện tích tỉ lệ với số tin đăng.</p>
            {selectedItem ? (
              <div className="rounded-md border p-2" role="status">
                <p className="font-medium">{selectedItem.name}</p>
                <p>{formatInt(selectedItem.n_listings)} tin</p>
                <p>{formatPriceM2(selectedItem.median_price_m2_mil)}</p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Bấm vào một vùng để xem số liệu.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {sorted.length > 0 ? (
        <DataTable
          caption="Số liệu theo khu vực"
          columns={["Khu vực", "Số tin", "Giá/m² trung vị", "Có toạ độ"]}
          rows={sorted.map((i) => [
            i.name,
            formatInt(i.n_listings),
            formatPriceM2(i.median_price_m2_mil),
            i.lat !== null ? "có" : "không",
          ])}
          className="max-h-96 overflow-y-auto"
        />
      ) : null}
    </div>
  );
}
