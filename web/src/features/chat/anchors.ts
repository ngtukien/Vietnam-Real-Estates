/** id DOM của thẻ nguồn; `scope` tách các cột ở trang so sánh (cùng tin xuất hiện ở nhiều cột). */
export function sourceAnchorId(scope: string, listingId: number): string {
  return `${scope}src-${listingId}`;
}
