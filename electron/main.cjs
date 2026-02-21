const { app, BrowserWindow, ipcMain, dialog, shell, safeStorage } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');

const DEFAULT_PORT = 8765;
const APP_CONFIG_FILE = 'app-config.json';
const SECURE_CONFIG_FILE = 'secure-config.json';
const WORKSPACE_ROOT = path.resolve(__dirname, '..');
const RUNTIME_ROOT = path.join(WORKSPACE_ROOT, 'runtime');
const CONFIG_ROOT = path.join(RUNTIME_ROOT, 'config');
const OUTPUT_ROOT = path.join(RUNTIME_ROOT, 'outputs');
const MODEL_ROOT = path.join(RUNTIME_ROOT, 'models');
const TEMP_ROOT = path.join(RUNTIME_ROOT, 'tmp');
const HF_HOME = path.join(MODEL_ROOT, 'hf-home');
const HF_HUB_CACHE = path.join(HF_HOME, 'hub');
const WHISPER_CACHE = path.join(MODEL_ROOT, 'whisper');
const ALIGN_MODEL_DIR = path.join(MODEL_ROOT, 'whisperx');
const VENV_PYTHON = path.join(RUNTIME_ROOT, '.venv', 'bin', 'python3');
const ALIGN_VENV_PYTHON = path.join(RUNTIME_ROOT, '.venv-align', 'bin', 'python3');

let mainWindow;
let pyProcess;
let backendUrl = `http://127.0.0.1:${DEFAULT_PORT}`;

function getConfigPath(fileName) {
  fs.mkdirSync(CONFIG_ROOT, { recursive: true });
  return path.join(CONFIG_ROOT, fileName);
}

function readJson(filePath, fallback) {
  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), 'utf-8');
}

function getAppConfig() {
  const configPath = getConfigPath(APP_CONFIG_FILE);
  const defaults = {
    locale: 'de',
    outputDir: OUTPUT_ROOT,
    modelCacheDir: MODEL_ROOT,
    performanceProfile: 'standard',
    qualityPreset: 'balanced',
    allowFallback: false
  };
  const current = readJson(configPath, {});
  return {
    ...defaults,
    locale: current.locale === 'en' ? 'en' : 'de',
    performanceProfile: current.performanceProfile === 'memory' ? 'memory' : 'standard',
    qualityPreset: ['speed', 'balanced', 'quality'].includes(current.qualityPreset) ? current.qualityPreset : 'balanced',
    allowFallback: current.allowFallback === true
  };
}

function setAppConfig(nextConfig) {
  const configPath = getConfigPath(APP_CONFIG_FILE);
  const persisted = {
    locale: nextConfig.locale === 'en' ? 'en' : 'de',
    performanceProfile: nextConfig.performanceProfile === 'memory' ? 'memory' : 'standard',
    qualityPreset: ['speed', 'balanced', 'quality'].includes(nextConfig.qualityPreset) ? nextConfig.qualityPreset : 'balanced',
    allowFallback: nextConfig.allowFallback === true
  };
  writeJson(configPath, persisted);
}

function getOrCreateVoiceSecret() {
  const securePath = getConfigPath(SECURE_CONFIG_FILE);
  const existing = readJson(securePath, null);

  if (existing?.encrypted && safeStorage.isEncryptionAvailable()) {
    try {
      const decrypted = safeStorage.decryptString(Buffer.from(existing.encrypted, 'base64'));
      if (decrypted) {
        return decrypted;
      }
    } catch {
      // fallthrough to regenerate if keychain changed
    }
  }

  if (existing?.plain) {
    return existing.plain;
  }

  const secret = crypto.randomBytes(32).toString('base64');
  if (safeStorage.isEncryptionAvailable()) {
    const encrypted = safeStorage.encryptString(secret).toString('base64');
    writeJson(securePath, { encrypted, version: 1 });
  } else {
    writeJson(securePath, { plain: secret, version: 1 });
  }
  return secret;
}

async function waitForHealth(url, timeoutMs = 20000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(`${url}/health`);
      if (res.ok) {
        return;
      }
    } catch {
      // retry until timeout
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error('Python service did not become healthy in time.');
}

async function startPythonService() {
  const config = getAppConfig();
  fs.mkdirSync(config.outputDir, { recursive: true });
  fs.mkdirSync(config.modelCacheDir, { recursive: true });
  fs.mkdirSync(TEMP_ROOT, { recursive: true });
  fs.mkdirSync(HF_HOME, { recursive: true });
  fs.mkdirSync(HF_HUB_CACHE, { recursive: true });
  fs.mkdirSync(WHISPER_CACHE, { recursive: true });
  fs.mkdirSync(ALIGN_MODEL_DIR, { recursive: true });

  const port = process.env.TTS_PORT || `${DEFAULT_PORT}`;
  backendUrl = `http://127.0.0.1:${port}`;

  const scriptPath = path.join(__dirname, '..', 'python_service', 'run.py');
  const voiceSecret = getOrCreateVoiceSecret();
  const pythonBin = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3';
  const alignPythonBin = ALIGN_VENV_PYTHON;

  pyProcess = spawn(pythonBin, [scriptPath, '--port', port], {
    cwd: WORKSPACE_ROOT,
    env: {
      ...process.env,
      TTS_PORT: port,
      TTS_MODEL_CACHE_DIR: config.modelCacheDir,
      TTS_OUTPUT_DIR: config.outputDir,
      TTS_VOICE_SECRET: voiceSecret,
      TTS_PERFORMANCE_PROFILE: config.performanceProfile,
      TTS_QUALITY_PRESET: config.qualityPreset,
      TTS_ALLOW_MACOS_FALLBACK: config.allowFallback ? '1' : '0',
      TTS_TMP_DIR: TEMP_ROOT,
      TTS_WHISPER_CACHE_DIR: WHISPER_CACHE,
      TTS_WHISPER_MODEL: 'tiny',
      TTS_ALIGN_PYTHON: alignPythonBin,
      TTS_ALIGN_MODEL_DIR: ALIGN_MODEL_DIR,
      TTS_ALIGN_LANGUAGES: 'de,en',
      TTS_ALIGN_MIN_COVERAGE: '0.97',
      HF_HOME,
      HUGGINGFACE_HUB_CACHE: HF_HUB_CACHE,
      TRANSFORMERS_CACHE: HF_HUB_CACHE,
      TMPDIR: TEMP_ROOT,
      TMP: TEMP_ROOT,
      TEMP: TEMP_ROOT
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  pyProcess.stdout.on('data', (chunk) => {
    const message = chunk.toString();
    process.stdout.write(`[py] ${message}`);
  });

  pyProcess.stderr.on('data', (chunk) => {
    const message = chunk.toString();
    process.stderr.write(`[py:err] ${message}`);
  });

  pyProcess.on('exit', (code) => {
    if (code !== 0) {
      console.error(`Python service exited with code ${code}`);
    }
  });

  await waitForHealth(backendUrl);
}

function stopPythonService() {
  if (!pyProcess) {
    return;
  }
  pyProcess.kill('SIGTERM');
  pyProcess = null;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1460,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#f5f7f2',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL;
  if (devServerUrl) {
    mainWindow.loadURL(devServerUrl);
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'renderer', 'index.html'));
  }
}

app.whenReady().then(async () => {
  await startPythonService();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonService();
});

ipcMain.handle('service:get-url', async () => backendUrl);

ipcMain.handle('config:get', async () => {
  return getAppConfig();
});

ipcMain.handle('config:set', async (_event, partialConfig) => {
  const current = getAppConfig();
  const next = {
    ...current,
    locale: partialConfig.locale ?? current.locale,
    performanceProfile: partialConfig.performanceProfile ?? current.performanceProfile,
    qualityPreset: partialConfig.qualityPreset ?? current.qualityPreset,
    allowFallback: partialConfig.allowFallback ?? current.allowFallback,
    outputDir: OUTPUT_ROOT,
    modelCacheDir: MODEL_ROOT
  };
  setAppConfig(next);
  stopPythonService();
  await startPythonService();
  return next;
});

ipcMain.handle('dialog:pick-pdf', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'PDF', extensions: ['pdf'] }]
  });

  if (result.canceled || !result.filePaths.length) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle('dialog:pick-audio', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'm4a', 'flac'] }]
  });

  if (result.canceled || !result.filePaths.length) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle('shell:open-path', async (_event, targetPath) => {
  return shell.openPath(targetPath);
});

ipcMain.handle('paths:get', async () => {
  const cfg = getAppConfig();
  return {
    userData: app.getPath('userData'),
    downloads: app.getPath('downloads'),
    workspaceRoot: WORKSPACE_ROOT,
    runtimeRoot: RUNTIME_ROOT,
    configRoot: CONFIG_ROOT,
    tempDir: TEMP_ROOT,
    outputDir: cfg.outputDir,
    modelCacheDir: cfg.modelCacheDir
  };
});
