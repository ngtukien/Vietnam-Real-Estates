import { useId, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { errorMessage } from "@/api/client";
import { useFilterOptions, useMarket, useOverview } from "@/api/queries";
import { PROPERTY_TYPES, type PropertyType } from "@/api/schema";
import { Alert, Card, CardContent, CardHeader, CardTitle, DataTable, Label, NativeSelect, Spinner } from "@/components/ui/primitives";
import { formatDecimal, formatInt, formatMonth, formatPriceM2 } from "@/lib/format";

const AXIS = { fontSize: 12, fill: "var(--muted-foreground)" };

function Overview() {
  const { data, error } = useOverview();
  if (error) return <Alert tone="error">{errorMessage(error)}</Alert>;
  if (!data) return <Spinner />;
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <Card>
        <CardContent className="pt-4">
          <p className="text-xs text-muted-foreground">Tin sau khử trùng lặp</p>
          <p className="text-2xl font-semibold">{formatInt(data.n_listings)}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-4">
          <p className="text-xs text-muted-foreground">Dòng thô</p>
          <p className="text-2xl font-semibold">{formatInt(data.n_raw_rows)}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-4">
          <p className="text-xs text-muted-foreground">Bản trùng đã loại</p>
          <p className="text-2xl font-semibold">{formatInt(data.n_duplicates_removed)}</p>
        </CardContent>
      </Card>
    </div>
  );
}

export default function MarketPage() {
  const id = useId();
  const [province, setProvince] = useState<string | null>(null);
  const [district, setDistrict] = useState<string | null>(null);
  const [propertyType, setPropertyType] = useState<PropertyType | null>(null);
  const { data: options } = useFilterOptions();
  const defaultProvince = options?.provinces.includes("Hồ Chí Minh") ? "Hồ Chí Minh" : (options?.provinces[0] ?? null);
  const effectiveProvince = province ?? defaultProvince;
  const { data, isLoading, error } = useMarket(effectiveProvince, district, propertyType);
  const districts = effectiveProvince ? (options?.districts[effectiveProvince] ?? []) : [];
  const byArea = [...(data?.by_area ?? [])]
    .filter((a) => a.median_price_m2_mil !== null)
    .sort((a, b) => (b.median_price_m2_mil ?? 0) - (a.median_price_m2_mil ?? 0))
    .slice(0, 25);
  const monthly = (data?.monthly ?? []).map((m) => ({ ...m, label: formatMonth(m.month) }));

  return (
    <div className="grid gap-4">
      <h1 className="text-lg font-semibold">Dashboard thị trường</h1>
      <Overview />
      <div className="flex flex-wrap items-end gap-4">
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-p`}>Tỉnh/thành</Label>
          <NativeSelect
            id={`${id}-p`}
            className="w-56"
            value={effectiveProvince ?? ""}
            onChange={(e) => {
              setProvince(e.target.value || null);
              setDistrict(null);
            }}
          >
            {options?.provinces.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </NativeSelect>
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-d`}>Quận/huyện</Label>
          <NativeSelect id={`${id}-d`} className="w-56" value={district ?? ""} onChange={(e) => setDistrict(e.target.value || null)}>
            <option value="">(cả tỉnh)</option>
            {districts.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </NativeSelect>
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-t`}>Loại BĐS</Label>
          <NativeSelect
            id={`${id}-t`}
            className="w-56"
            value={propertyType ?? ""}
            onChange={(e) => setPropertyType((e.target.value || null) as PropertyType | null)}
          >
            <option value="">(tất cả)</option>
            {PROPERTY_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </NativeSelect>
        </div>
        {isLoading ? <Spinner /> : null}
      </div>
      {error ? <Alert tone="error">{errorMessage(error)}</Alert> : null}

      {data ? (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Giá/m² trung vị theo {data.scope.district ? "phường/xã" : "quận/huyện"}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[420px]" aria-hidden data-testid="chart-by-area">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart accessibilityLayer={false} data={byArea} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tick={AXIS} tickFormatter={(v: number) => formatDecimal(v, 0)} />
                    <YAxis type="category" dataKey="name" width={120} tick={AXIS} interval={0} />
                    <Tooltip formatter={(v) => formatPriceM2(Number(v))} />
                    <Bar dataKey="median_price_m2_mil" name="Giá/m²" fill="var(--chart-1)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <details className="mt-2 text-sm">
                <summary className="cursor-pointer">Bảng số liệu</summary>
                <DataTable
                  caption="Giá/m² theo khu vực"
                  columns={["Khu vực", "Số tin", "Giá/m² trung vị"]}
                  rows={byArea.map((a) => [a.name, formatInt(a.n), formatPriceM2(a.median_price_m2_mil)])}
                  className="mt-2"
                />
              </details>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Xu hướng theo tháng</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[420px]" aria-hidden data-testid="chart-monthly">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart accessibilityLayer={false} data={monthly} margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" tick={AXIS} />
                    <YAxis tick={AXIS} tickFormatter={(v: number) => formatDecimal(v, 0)} />
                    <Tooltip formatter={(v) => formatPriceM2(Number(v))} />
                    <Line type="monotone" dataKey="median_price_m2_mil" name="Giá/m²" stroke="var(--chart-2)" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <details className="mt-2 text-sm">
                <summary className="cursor-pointer">Bảng số liệu</summary>
                <DataTable
                  caption="Xu hướng theo tháng"
                  columns={["Tháng", "Số tin", "Giá/m² trung vị"]}
                  rows={monthly.map((m) => [m.label, formatInt(m.n), formatPriceM2(m.median_price_m2_mil)])}
                  className="mt-2"
                />
              </details>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
