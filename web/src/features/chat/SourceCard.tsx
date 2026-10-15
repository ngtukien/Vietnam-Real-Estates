import { Network, Quote } from "lucide-react";
import type { ListingCard } from "@/api/schema";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/primitives";
import { formatArea, formatMonth, formatPriceM2, formatVnd } from "@/lib/format";
import { cn } from "@/lib/utils";
import { sourceAnchorId } from "./anchors";

export function SourceCard({
  card,
  scope = "",
  onShowGraph,
}: {
  card: ListingCard;
  scope?: string;
  onShowGraph?: (listingId: number) => void;
}) {
  const a = card.address;
  const place = [a.street, a.ward, a.district, a.province].filter(Boolean).join(", ");
  return (
    <article
      id={sourceAnchorId(scope, card.listing_id)}
      tabIndex={-1}
      aria-labelledby={`${sourceAnchorId(scope, card.listing_id)}-title`}
      className={cn(
        "rounded-md border bg-card p-3 text-sm scroll-mt-24 focus:ring-2 focus:ring-ring",
        card.cited && "border-primary",
      )}
      data-testid="source-card"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="outline" className="font-mono">
          Tin#{card.listing_id}
        </Badge>
        {card.cited ? (
          <Badge tone="default">
            <Quote className="size-3" aria-hidden /> được trích
          </Badge>
        ) : null}
        <Badge>{card.property_type}</Badge>
        {card.deal_type === "rent" ? <Badge tone="warning">cho thuê</Badge> : null}
      </div>
      <h3 id={`${sourceAnchorId(scope, card.listing_id)}-title`} className="mt-2 font-medium leading-snug">
        {card.title ?? "(không có tiêu đề)"}
      </h3>
      <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs text-muted-foreground sm:grid-cols-4">
        <div>
          <dt className="sr-only">Giá</dt>
          <dd className="font-medium text-foreground">{formatVnd(card.price_vnd)}</dd>
        </div>
        <div>
          <dt className="sr-only">Giá/m²</dt>
          <dd>{formatPriceM2(card.price_m2_mil)}</dd>
        </div>
        <div>
          <dt className="sr-only">Diện tích</dt>
          <dd>{formatArea(card.area_m2)}</dd>
        </div>
        <div>
          <dt className="sr-only">Tháng đăng</dt>
          <dd>{formatMonth(card.published_month)}</dd>
        </div>
      </dl>
      <p className="mt-1 text-xs text-muted-foreground">
        {place || "(không rõ địa chỉ)"}
        {card.project ? ` · Dự án ${card.project}` : ""}
        {card.bedrooms !== null ? ` · ${card.bedrooms} PN` : ""}
      </p>
      {card.description_excerpt ? <p className="mt-2 line-clamp-3 text-xs">{card.description_excerpt}</p> : null}
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-xs text-muted-foreground">
          {card.score !== null ? `điểm ${card.score.toFixed(3)}` : "từ Cypher"}
        </span>
        {onShowGraph ? (
          <Button size="sm" variant="outline" onClick={() => onShowGraph(card.listing_id)}>
            <Network aria-hidden /> Đồ thị
          </Button>
        ) : null}
      </div>
    </article>
  );
}
