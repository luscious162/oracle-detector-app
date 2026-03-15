#!/usr/bin/env python3
"""
甲骨文智能分析系统 - 一键启动脚本
同时启动 Flask 后端和 React 前端
"""

import os
import sys
import subprocess
import time
import threading
import signal
import platform
import webbrowser
import socket

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    color = colors.get(status, Colors.BLUE)
    print(f"{color}{message}{Colors.RESET}")

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def find_available_port(start_port=5000, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port

def check_node_installed():
    try:
        subprocess.run(['node', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_npm_installed():
    try:
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_frontend_dependencies(frontend_dir):
    print_status("正在安装前端依赖...", "info")
    try:
        subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
        print_status("前端依赖安装完成", "success")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"前端依赖安装失败: {e}", "error")
        return False

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_frontend(frontend_dir):
    try:
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=None if platform.system() != 'Windows' else None
        )
        return process
    except Exception as e:
        print_status(f"前端启动失败: {e}", "error")
        return None

def start_backend(backend_dir, port=5000):
    print_status(f"正在启动后端 API 服务器 (端口 {port})...", "info")
    try:
        env = os.environ.copy()
        env['FLASK_RUN_PORT'] = str(port)
        process = subprocess.Popen(
            [sys.executable, 'app.py', '--port', str(port)],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=None if platform.system() != 'Windows' else None,
            env=env
        )
        return process, port
    except Exception as e:
        print_status(f"后端启动失败: {e}", "error")
        return None, port

def read_output(process, name):
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                if name == 'Backend':
                    print(f"{Colors.YELLOW}[后端]{Colors.RESET} {line.rstrip()}")
                else:
                    print(f"{Colors.BLUE}[前端]{Colors.RESET} {line.rstrip()}")
    except:
        pass

def open_browser(url, delay=3):
    """延迟打开浏览器"""
    time.sleep(delay)
    webbrowser.open(url)

def main():
    project_root = get_project_root()
    frontend_dir = project_root
    backend_dir = project_root

    print(f"\n{Colors.BOLD}{'='*50}")
    print("  甲骨文智能分析全能系统 - 一键启动")
    print(f"{'='*50}{Colors.RESET}\n")

    if not check_node_installed():
        print_status("Node.js 未安装，请先安装 Node.js", "error")
        print_status("   下载地址: https://nodejs.org/", "info")
        sys.exit(1)

    if not check_npm_installed():
        print_status("npm 未安装，请先安装 Node.js", "error")
        sys.exit(1)

    node_version = subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()
    npm_version = subprocess.run(['npm', '--version'], capture_output=True, text=True).stdout.strip()
    print_status(f"Node.js: {node_version}", "success")
    print_status(f"npm: {npm_version}", "success")

    node_modules = os.path.join(frontend_dir, 'node_modules')
    if not os.path.exists(node_modules):
        print_status("前端依赖未安装，开始安装...", "warning")
        if not install_frontend_dependencies(frontend_dir):
            sys.exit(1)
    else:
        print_status("前端依赖已安装", "success")

    # 查找可用端口
    backend_port = find_available_port(5000)
    if backend_port != 5000:
        print_status(f"端口 5000 被占用，使用端口 {backend_port}", "warning")
    
    # 将后端端口写入文件，供 Vite 读取
    port_file = os.path.join(frontend_dir, '.backend_port')
    with open(port_file, 'w') as f:
        f.write(str(backend_port))

    processes = []

    try:
        backend_process, used_port = start_backend(backend_dir, backend_port)
        if backend_process:
            processes.append(('Backend', backend_process))
            time.sleep(2)
        else:
            print_status("后端启动失败", "error")
            sys.exit(1)

        frontend_process = start_frontend(frontend_dir)
        if frontend_process:
            processes.append(('Frontend', frontend_process))
        else:
            print_status("前端启动失败", "error")
            sys.exit(1)

        print(f"\n{Colors.BOLD}{'='*50}")
        print("  启动成功！")
        print(f"{'='*50}{Colors.RESET}\n")

        print_status("服务地址:", "info")
        print(f"   {Colors.BLUE}前端: http://localhost:3000{Colors.RESET}")
        print(f"   {Colors.BLUE}后端: http://127.0.0.1:{used_port}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}按 Ctrl+C 停止所有服务{Colors.RESET}\n")

        # 自动打开浏览器
        browser_thread = threading.Thread(target=open_browser, args=('http://localhost:3000', 3))
        browser_thread.daemon = True
        browser_thread.start()

        threads = []
        for name, proc in processes:
            t = threading.Thread(target=read_output, args=(proc, name))
            t.daemon = True
            t.start()
            threads.append(t)

        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print_status(f"{name} 进程已退出", "error")
                    sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}正在停止服务...{Colors.RESET}")
        for name, process in processes:
            print_status(f"   停止 {name}...", "warning")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print_status("所有服务已停止", "success")
        sys.exit(0)

if __name__ == '__main__':
    main()
