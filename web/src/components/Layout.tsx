import { Building2, KeyRound, Moon, Sun } from "lucide-react";
import { Suspense, useState } from "react";
import { NavLink, Outlet } from "react-router";
import { getApiKey, setApiKey } from "@/api/client";
import { useHealth } from "@/api/queries";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Badge, Input, Label, Spinner } from "@/components/ui/primitives";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Hỏi đáp" },
  { to: "/compare", label: "So sánh 4 hệ" },
  { to: "/map", label: "Bản đồ" },
  { to: "/valuate", label: "Định giá" },
  { to: "/market", label: "Thị trường" },
  { to: "/benchmark", label: "Benchmark" },
] as const;

function SettingsDialog() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(() => Boolean(getApiKey()));
  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (o) setValue(getApiKey() ?? "");
      }}
    >
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" aria-label="Cài đặt API key">
          <KeyRound aria-hidden />
          <span className="hidden sm:inline">API key</span>
          {saved ? <span className="size-2 rounded-full bg-success" aria-label="đã nhập key" /> : null}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Cài đặt</DialogTitle>
        <DialogDescription>
          Nếu máy chủ bật <code>VRE_API_TOKEN</code>, nhập key ở đây. Key chỉ lưu trong phiên trình duyệt này
          (sessionStorage) và gửi qua header <code>X-API-Key</code>, không bao giờ nằm trên URL.
        </DialogDescription>
        <form
          className="grid gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            setApiKey(value.trim() || null);
            setSaved(Boolean(value.trim()));
            setOpen(false);
          }}
        >
          <div className="grid gap-1.5">
            <Label htmlFor="api-key">API key</Label>
            <Input
              id="api-key"
              type="password"
              autoComplete="off"
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => {
                setApiKey(null);
                setValue("");
                setSaved(false);
              }}
            >
              Xoá key
            </Button>
            <Button type="submit">Lưu</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function HealthBadge() {
  const { data, isError } = useHealth();
  if (isError) return <Badge tone="destructive">API mất kết nối</Badge>;
  if (!data) return null;
  return data.status === "ok" ? (
    <Badge tone="outline">API sẵn sàng</Badge>
  ) : (
    <Badge tone="warning" title={`Qdrant: ${data.qdrant ? "ok" : "lỗi"} · Neo4j: ${data.neo4j ? "ok" : "lỗi"}`}>
      API suy giảm
    </Badge>
  );
}

export function Layout() {
  const [theme, toggleTheme] = useTheme();
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-background focus:px-3 focus:py-2"
      >
        Bỏ qua tới nội dung
      </a>
      <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-2">
          <NavLink to="/" className="flex items-center gap-2 font-semibold">
            <Building2 className="size-5 text-primary" aria-hidden />
            <span>VRE · BĐS Việt Nam</span>
          </NavLink>
          <nav aria-label="Trang chính" className="order-3 w-full overflow-x-auto md:order-none md:w-auto">
            <ul className="flex gap-1">
              {NAV.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.to === "/"}
                    className={({ isActive }) =>
                      cn(
                        "block whitespace-nowrap rounded-md px-3 py-1.5 text-sm",
                        isActive ? "bg-accent font-medium text-accent-foreground" : "hover:bg-muted",
                      )
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <HealthBadge />
            <SettingsDialog />
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
            >
              {theme === "dark" ? <Sun aria-hidden /> : <Moon aria-hidden />}
            </Button>
          </div>
        </div>
      </header>
      <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 py-4" tabIndex={-1}>
        <Suspense fallback={<Spinner />}>
          <Outlet />
        </Suspense>
      </main>
      <footer className="border-t text-xs text-muted-foreground">
        <div className="mx-auto max-w-7xl space-y-1 px-4 py-3">
          <p>
            Dữ liệu tin đăng: <span className="font-medium">tinixai/vietnam-real-estates</span> (CC BY-NC 4.0, chỉ dùng phi
            thương mại). Số liệu chỉ mang tính tham khảo, không phải tư vấn đầu tư.
          </p>
          <p>
            Ranh giới hành chính (trước sáp nhập 2025):{" "}
            <a className="underline" href="https://www.geoboundaries.org/" rel="noreferrer" target="_blank">
              geoBoundaries
            </a>{" "}
            gbOpen VNM — ADM2: OCHA ROAP (CC BY 3.0 IGO), ADM1: Public Domain. Ranh tỉnh được ghép từ ranh quận/huyện.
          </p>
        </div>
      </footer>
    </div>
  );
}
