const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1700,
    height: 842,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.setMenu(null);
  mainWindow.loadURL('http://localhost:5173');

  // 봇 실행
  const bot = spawn('python', [path.join(__dirname, 'bot', 'new_tradeBot.py')]);

  // 로그 전송
  bot.stdout.on('data', (data) => {
    mainWindow.webContents.send('bot-log', data.toString());
  });

  bot.stderr.on('data', (data) => {
    mainWindow.webContents.send('bot-log', `⚠️ ${data.toString()}`);
  });

  bot.on('exit', (code) => {
    mainWindow.webContents.send('bot-log', `✅ 봇 종료됨 (코드: ${code})`);
  });
}

app.whenReady().then(() => {
  createWindow();
});