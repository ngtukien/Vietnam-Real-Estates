import { Link, useRouteError } from "react-router";

export function RouteError({ notFound = false }: { notFound?: boolean }) {
  const error = useRouteError();
  return (
    <div className="mx-auto max-w-xl p-8">
      <h1 className="text-xl font-semibold">{notFound ? "Không tìm thấy trang" : "Đã có lỗi khi hiển thị trang"}</h1>
      {!notFound && error instanceof Error ? <p className="mt-2 text-sm text-muted-foreground">{error.message}</p> : null}
      <Link to="/" className="mt-4 inline-block text-primary underline">
        Về trang hỏi đáp
      </Link>
    </div>
  );
}
