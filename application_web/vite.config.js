import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// 读取后端端口配置
let backendPort = 5000
try {
  const portFile = path.resolve(process.cwd(), '.backend_port')
  if (fs.existsSync(portFile)) {
    backendPort = parseInt(fs.readFileSync(portFile, 'utf-8').trim())
  }
} catch (e) {
  // 使用默认端口
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
