/// <reference types="electron-vite/node" />

// Variables the main process may read (electron-vite exposes only the MAIN_VITE_ prefix).
interface ImportMetaEnv {
  /** Python used to run the finger-tapping demo; the repository's .venv when unset. */
  readonly MAIN_VITE_TAPPING_PYTHON?: string;
}
