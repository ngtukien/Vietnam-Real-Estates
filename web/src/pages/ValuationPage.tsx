import { useMutation } from "@tanstack/react-query";
import { Calculator } from "lucide-react";
import { useId, useState } from "react";
import { apiPost, errorMessage } from "@/api/client";
import { useFilterOptions } from "@/api/queries";
import { PROPERTY_TYPES, type PropertyType, type ValuationInput, type ValuationResult } from "@/api/schema";
import { Button } from "@/components/ui/button";
import { Alert, Card, CardContent, CardHeader, CardTitle, Input, Label, NativeSelect, Spinner } from "@/components/ui/primitives";
import { ValuationResultView } from "@/features/valuation/ValuationResultView";
import { formatMonth, parseOptionalNumber } from "@/lib/format";

const DIRECTIONS = ["Đông", "Tây", "Nam", "Bắc", "Đông Bắc", "Đông Nam", "Tây Bắc", "Tây Nam"];

interface FormState {
  property_type: PropertyType;
  province: string;
  district: string;
  ward: string;
  area_m2: string;
  bedrooms: string;
  bathrooms: string;
  floors: string;
  frontage_m: string;
  road_width_m: string;
  house_direction: string;
  month: string;
  asking_price_bil: string;
}

const INITIAL: FormState = {
  property_type: "Căn hộ chung cư",
  province: "",
  district: "",
  ward: "",
  area_m2: "",
  bedrooms: "",
  bathrooms: "",
  floors: "",
  frontage_m: "",
  road_width_m: "",
  house_direction: "",
  month: "",
  asking_price_bil: "",
};

function toInput(f: FormState): ValuationInput {
  const asking = parseOptionalNumber(f.asking_price_bil);
  return {
    property_type: f.property_type,
    deal_type: "sale",
    province: f.province,
    district: f.district,
    ward: f.ward.trim() || null,
    area_m2: parseOptionalNumber(f.area_m2) ?? 0,
    bedrooms: parseOptionalNumber(f.bedrooms),
    bathrooms: parseOptionalNumber(f.bathrooms),
    floors: parseOptionalNumber(f.floors),
    frontage_m: parseOptionalNumber(f.frontage_m),
    road_width_m: parseOptionalNumber(f.road_width_m),
    house_direction: f.house_direction || null,
    month: f.month || null,
    amenities: [],
    asking_price_vnd: asking === null ? null : asking * 1e9,
  };
}

export default function ValuationPage() {
  const id = useId();
  const [form, setForm] = useState<FormState>(INITIAL);
  const { data: options } = useFilterOptions();
  const mutation = useMutation({ mutationFn: (input: ValuationInput) => apiPost<ValuationResult>("/api/valuate", input) });
  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value, ...(k === "province" ? { district: "" } : {}) }));
  const area = parseOptionalNumber(form.area_m2);
  const areaInvalid = form.area_m2 !== "" && (area === null || area <= 0 || area > 100000);
  const canSubmit = form.province && form.district && area !== null && !areaInvalid;
  const districts = form.province ? (options?.districts[form.province] ?? []) : [];

  const field = (key: keyof FormState, label: string, extra?: React.InputHTMLAttributes<HTMLInputElement>) => (
    <div className="grid gap-1.5">
      <Label htmlFor={`${id}-${key}`}>{label}</Label>
      <Input id={`${id}-${key}`} inputMode="decimal" value={form[key]} onChange={set(key)} {...extra} />
    </div>
  );

  return (
    <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
      <Card>
        <CardHeader>
          <CardTitle as="h1">Định giá bất động sản</CardTitle>
          <p className="text-xs text-muted-foreground">Module phụ: XGBoost trên dữ liệu tin đăng (giá chào, không phải giá giao dịch).</p>
        </CardHeader>
        <CardContent>
          <form
            className="grid gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (canSubmit) mutation.mutate(toInput(form));
            }}
          >
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-type`}>Loại BĐS</Label>
              <NativeSelect id={`${id}-type`} value={form.property_type} onChange={set("property_type")}>
                {PROPERTY_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </NativeSelect>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-province`}>Tỉnh/thành *</Label>
              <NativeSelect id={`${id}-province`} value={form.province} onChange={set("province")} required>
                <option value="">(chọn)</option>
                {options?.provinces.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </NativeSelect>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-district`}>Quận/huyện *</Label>
              <NativeSelect
                id={`${id}-district`}
                value={form.district}
                onChange={set("district")}
                disabled={!form.province}
                required
              >
                <option value="">(chọn)</option>
                {districts.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </NativeSelect>
            </div>
            {field("ward", "Phường/xã", { inputMode: "text" })}
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-area_m2`}>Diện tích (m²) *</Label>
              <Input
                id={`${id}-area_m2`}
                inputMode="decimal"
                value={form.area_m2}
                onChange={set("area_m2")}
                required
                aria-invalid={areaInvalid}
                aria-describedby={areaInvalid ? `${id}-area-err` : undefined}
              />
              {areaInvalid ? (
                <p id={`${id}-area-err`} className="text-xs text-destructive">
                  Diện tích phải lớn hơn 0 và không quá 100.000 m².
                </p>
              ) : null}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {field("bedrooms", "Phòng ngủ")}
              {field("bathrooms", "Phòng tắm")}
              {field("floors", "Số tầng")}
              {field("frontage_m", "Mặt tiền (m)")}
              {field("road_width_m", "Đường vào (m)")}
              <div className="grid gap-1.5">
                <Label htmlFor={`${id}-dir`}>Hướng nhà</Label>
                <NativeSelect id={`${id}-dir`} value={form.house_direction} onChange={set("house_direction")}>
                  <option value="">(không rõ)</option>
                  {DIRECTIONS.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </NativeSelect>
              </div>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor={`${id}-month`}>Tháng định giá</Label>
              <NativeSelect id={`${id}-month`} value={form.month} onChange={set("month")}>
                <option value="">(mới nhất trong dữ liệu)</option>
                {options?.months.map((m) => (
                  <option key={m} value={m}>
                    {formatMonth(m)}
                  </option>
                ))}
              </NativeSelect>
            </div>
            {field("asking_price_bil", "Giá chào (tỷ đồng) — để nhận định")}
            <Button type="submit" disabled={!canSubmit || mutation.isPending}>
              <Calculator aria-hidden /> Định giá
            </Button>
          </form>
        </CardContent>
      </Card>
      <div className="grid content-start gap-4" aria-live="polite">
        {mutation.isPending ? <Spinner label="Đang định giá" /> : null}
        {mutation.error ? <Alert tone="error">{errorMessage(mutation.error)}</Alert> : null}
        {mutation.data ? (
          <Card>
            <CardContent className="pt-4">
              <ValuationResultView result={mutation.data} />
            </CardContent>
          </Card>
        ) : !mutation.isPending ? (
          <p className="text-sm text-muted-foreground">Điền tỉnh, quận và diện tích để định giá.</p>
        ) : null}
      </div>
    </div>
  );
}
