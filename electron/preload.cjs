const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopApi', {
  getServiceUrl: () => ipcRenderer.invoke('service:get-url'),
  getConfig: () => ipcRenderer.invoke('config:get'),
  setConfig: (partialConfig) => ipcRenderer.invoke('config:set', partialConfig),
  pickPdf: () => ipcRenderer.invoke('dialog:pick-pdf'),
  pickAudio: () => ipcRenderer.invoke('dialog:pick-audio'),
  openPath: (targetPath) => ipcRenderer.invoke('shell:open-path', targetPath),
  getPaths: () => ipcRenderer.invoke('paths:get')
});
