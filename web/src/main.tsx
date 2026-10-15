import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode, lazy } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import { ApiError } from "@/api/client";
import { Layout } from "@/components/Layout";
import { RouteError } from "@/components/RouteError";
import { applyTheme, initialTheme } from "@/lib/theme";
import ChatPage from "@/pages/ChatPage";
import "./index.css";

applyTheme(initialTheme());

const ComparePage = lazy(() => import("@/pages/ComparePage"));
const MapPage = lazy(() => import("@/pages/MapPage"));
const ValuationPage = lazy(() => import("@/pages/ValuationPage"));
const MarketPage = lazy(() => import("@/pages/MarketPage"));
const BenchmarkPage = lazy(() => import("@/pages/BenchmarkPage"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      refetchOnWindowFocus: false,
      // Lỗi 4xx (sai key, quá giới hạn, không có dữ liệu) không thử lại.
      retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
    },
  },
});

const router = createBrowserRouter([
  {
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <ChatPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "map", element: <MapPage /> },
      { path: "valuate", element: <ValuationPage /> },
      { path: "market", element: <MarketPage /> },
      { path: "benchmark", element: <BenchmarkPage /> },
      { path: "*", element: <RouteError notFound /> },
    ],
  },
]);

const root = document.getElementById("root");
if (!root) throw new Error("Thiếu #root trong index.html");
createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
