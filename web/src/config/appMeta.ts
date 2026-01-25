export const APP_NAME = (import.meta.env.VITE_APP_TITLE as string | undefined) || "可萌";

export const APP_VERSION = (import.meta.env.VITE_APP_VERSION as string | undefined) || "0.0.0";

export const BUILD_SHA = (import.meta.env.VITE_BUILD_SHA as string | undefined) || "";
export const BUILD_TIME = (import.meta.env.VITE_BUILD_TIME as string | undefined) || "";

export function getBuildLabel(): string {
  const parts: string[] = [];
  if (APP_VERSION) parts.push(`v${APP_VERSION}`);
  if (BUILD_SHA) parts.push(BUILD_SHA.slice(0, 7));
  return parts.join(" · ");
}

