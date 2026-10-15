import { useId } from "react";
import { useFilterOptions } from "@/api/queries";
import { EMPTY_FILTERS, PROPERTY_TYPES, type PropertyType, type QueryFilters } from "@/api/schema";
import { Button } from "@/components/ui/button";
import { Input, Label, NativeSelect } from "@/components/ui/primitives";
import { formatMonth, parseOptionalNumber } from "@/lib/format";

/** Bộ lọc chỉnh tay (api-contract §2/§4): trường khác null GHI ĐÈ trường LLM trích ra. */
export function FilterPanel({ value, onChange }: { value: QueryFilters; onChange: (f: QueryFilters) => void }) {
  const id = useId();
  const { data: options, isError } = useFilterOptions();
  const districts = value.province ? (options?.districts[value.province] ?? []) : [];
  const set = <K extends keyof QueryFilters>(key: K, v: QueryFilters[K]) => onChange({ ...value, [key]: v });
  const num = (key: "min_price_mil" | "max_price_mil" | "min_area_m2" | "max_area_m2" | "bedrooms") => ({
    id: `${id}-${key}`,
    inputMode: "decimal" as const,
    value: value[key] ?? "",
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => set(key, parseOptionalNumber(e.target.value)),
  });
  const active = Object.values(value).filter((v) => v !== null).length;

  return (
    <fieldset className="grid gap-3 rounded-md border p-3" aria-describedby={`${id}-hint`}>
      <legend className="px-1 text-sm font-medium">Bộ lọc chỉnh tay {active ? `(${active})` : ""}</legend>
      <p id={`${id}-hint`} className="text-xs text-muted-foreground">
        Để trống = để LLM tự trích từ câu hỏi. Ô đã điền sẽ ghi đè giá trị LLM trích.
        {isError ? " Không tải được danh sách tỉnh/quận." : ""}
      </p>
      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-province`}>Tỉnh/thành</Label>
        <NativeSelect
          id={`${id}-province`}
          value={value.province ?? ""}
          onChange={(e) => onChange({ ...value, province: e.target.value || null, district: null })}
        >
          <option value="">(tất cả)</option>
          {options?.provinces.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </NativeSelect>
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-district`}>Quận/huyện</Label>
        <NativeSelect
          id={`${id}-district`}
          value={value.district ?? ""}
          disabled={!value.province}
          onChange={(e) => set("district", e.target.value || null)}
        >
          <option value="">(tất cả)</option>
          {districts.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </NativeSelect>
      </div>
      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-type`}>Loại BĐS</Label>
        <NativeSelect
          id={`${id}-type`}
          value={value.property_type ?? ""}
          onChange={(e) => set("property_type", (e.target.value || null) as PropertyType | null)}
        >
          <option value="">(tất cả)</option>
          {(options?.property_types ?? PROPERTY_TYPES).map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </NativeSelect>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-min_price_mil`}>Giá từ (triệu)</Label>
          <Input {...num("min_price_mil")} />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-max_price_mil`}>Giá đến (triệu)</Label>
          <Input {...num("max_price_mil")} />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-min_area_m2`}>Diện tích từ (m²)</Label>
          <Input {...num("min_area_m2")} />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-max_area_m2`}>Diện tích đến (m²)</Label>
          <Input {...num("max_area_m2")} />
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-month_from`}>Từ tháng</Label>
          <NativeSelect
            id={`${id}-month_from`}
            value={value.month_from ?? ""}
            onChange={(e) => set("month_from", e.target.value || null)}
          >
            <option value="">(mọi)</option>
            {options?.months.map((m) => (
              <option key={m} value={m}>
                {formatMonth(m)}
              </option>
            ))}
          </NativeSelect>
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor={`${id}-month_to`}>Đến tháng</Label>
          <NativeSelect
            id={`${id}-month_to`}
            value={value.month_to ?? ""}
            onChange={(e) => set("month_to", e.target.value || null)}
          >
            <option value="">(mọi)</option>
            {options?.months.map((m) => (
              <option key={m} value={m}>
                {formatMonth(m)}
              </option>
            ))}
          </NativeSelect>
        </div>
      </div>
      <Button variant="outline" size="sm" onClick={() => onChange(EMPTY_FILTERS)} disabled={!active}>
        Xoá bộ lọc
      </Button>
    </fieldset>
  );
}

