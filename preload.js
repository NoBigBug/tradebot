// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onBotLog: (callback) => ipcRenderer.on('bot-log', (_, data) => callback(data)),
});