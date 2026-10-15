import { useQuery } from "@tanstack/react-query";
import { apiGet } from "./client";
import type {
  BenchmarkResponse,
  FilterOptions,
  GeoAreas,
  GeoLevel,
  HealthResponse,
  InfoResponse,
  ListingDetail,
  MarketStats,
  PropertyType,
  StatsOverview,
  Subgraph,
} from "./schema";

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: ({ signal }) => apiGet<HealthResponse>("/api/health", undefined, signal) });

export const useInfo = () =>
  useQuery({ queryKey: ["info"], queryFn: ({ signal }) => apiGet<InfoResponse>("/api/info", undefined, signal) });

export const useFilterOptions = () =>
  useQuery({
    queryKey: ["filter-options"],
    queryFn: ({ signal }) => apiGet<FilterOptions>("/api/filters/options", undefined, signal),
    staleTime: Infinity,
  });

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: ({ signal }) => apiGet<StatsOverview>("/api/stats/overview", undefined, signal) });

export const useMarket = (province: string | null, district: string | null, propertyType: PropertyType | null) =>
  useQuery({
    queryKey: ["market", province, district, propertyType],
    enabled: Boolean(province),
    queryFn: ({ signal }) =>
      apiGet<MarketStats>(
        "/api/stats/market",
        { province, district, property_type: propertyType, deal_type: "sale" },
        signal,
      ),
  });

export const useGeoAreas = (level: GeoLevel, province: string | null) =>
  useQuery({
    queryKey: ["geo", level, province],
    enabled: level === "province" || Boolean(province),
    queryFn: ({ signal }) =>
      apiGet<GeoAreas>("/api/geo/areas", { level, province: level === "province" ? null : province }, signal),
  });

export const useSubgraph = (listingId: number | null) =>
  useQuery({
    queryKey: ["subgraph", listingId],
    enabled: listingId !== null,
    queryFn: ({ signal }) => apiGet<Subgraph>("/api/graph/subgraph", { listing_id: listingId }, signal),
  });

export const useListing = (listingId: number | null) =>
  useQuery({
    queryKey: ["listing", listingId],
    enabled: listingId !== null,
    queryFn: ({ signal }) => apiGet<ListingDetail>(`/api/listings/${listingId}`, undefined, signal),
  });

export const useBenchmark = () =>
  useQuery({
    queryKey: ["benchmark"],
    queryFn: ({ signal }) => apiGet<BenchmarkResponse>("/api/benchmark", undefined, signal),
    retry: false,
  });
