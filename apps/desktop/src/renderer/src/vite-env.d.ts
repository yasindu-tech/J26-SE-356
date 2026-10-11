/// <reference types="vite/client" />

/** Injected at build time from apps/desktop/package.json — see electron.vite.config.ts. */
declare const __APP_VERSION__: string;

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
