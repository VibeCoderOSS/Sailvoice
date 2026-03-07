import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { createServer } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const require = createRequire(import.meta.url);
const electronBinary = require('electron');

let viteServer;
let electronProcess;
let shuttingDown = false;

function shutdown(exitCode = 0) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  const finalize = async () => {
    if (electronProcess && !electronProcess.killed) {
      electronProcess.kill('SIGTERM');
    }
    if (viteServer) {
      await viteServer.close();
    }
    process.exit(exitCode);
  };

  finalize().catch((error) => {
    console.error('[dev-launcher] shutdown failed', error);
    process.exit(exitCode || 1);
  });
}

async function main() {
  const preferredPort = Number(process.env.VITE_PORT || 5173);
  viteServer = await createServer({
    root: projectRoot,
    server: {
      host: '127.0.0.1',
      port: preferredPort,
      strictPort: false,
    },
  });

  await viteServer.listen();

  const devUrl =
    viteServer.resolvedUrls?.local?.[0]
    ?? viteServer.resolvedUrls?.network?.[0]
    ?? `http://127.0.0.1:${preferredPort}`;

  console.log(`[dev-launcher] Vite ready at ${devUrl}`);

  electronProcess = spawn(electronBinary, ['electron/main.cjs'], {
    cwd: projectRoot,
    env: {
      ...process.env,
      VITE_DEV_SERVER_URL: devUrl,
    },
    stdio: 'inherit',
  });

  electronProcess.on('exit', (code, signal) => {
    if (signal) {
      shutdown(0);
      return;
    }
    shutdown(code ?? 0);
  });
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

main().catch((error) => {
  console.error('[dev-launcher] startup failed', error);
  shutdown(1);
});
