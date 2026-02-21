import type { AppConfig } from './models';

declare global {
  interface Window {
    desktopApi: {
      getServiceUrl: () => Promise<string>;
      getConfig: () => Promise<AppConfig>;
      setConfig: (partialConfig: Partial<AppConfig>) => Promise<AppConfig>;
      pickPdf: () => Promise<string | null>;
      pickAudio: () => Promise<string | null>;
      openPath: (targetPath: string) => Promise<string>;
      getPaths: () => Promise<{
        userData: string;
        downloads: string;
        workspaceRoot: string;
        runtimeRoot: string;
        configRoot: string;
        tempDir: string;
        outputDir: string;
        modelCacheDir: string;
      }>;
    };
  }
}

export {};
