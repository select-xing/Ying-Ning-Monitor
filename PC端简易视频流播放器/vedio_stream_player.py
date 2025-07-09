import tkinter as tk
from tkinter import messagebox
import subprocess
import threading

class VideoStreamPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("简易流媒体播放器")
        
        # 播放器进程
        self.ffplay_process = None
        
        # 创建界面
        self.setup_ui()
    
    def setup_ui(self):
        # 输入框和标签
        tk.Label(self.root, text="流媒体地址:").pack(pady=5)
        self.url_entry = tk.Entry(self.root, width=50)
        self.url_entry.pack(pady=5)
        self.url_entry.insert(0, "rtmp://localhost/live/stream")  # 默认地址
        
        # 播放按钮
        self.play_btn = tk.Button(self.root, text="播放", command=self.start_playback)
        self.play_btn.pack(pady=10)
        
        # 停止按钮
        self.stop_btn = tk.Button(self.root, text="停止", command=self.stop_playback, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
    
    def start_playback(self):
        stream_url = self.url_entry.get().strip()
        if not stream_url:
            messagebox.showerror("错误", "请输入有效的流媒体地址")
            return
        
        # 禁用播放按钮，启用停止按钮
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 在新线程中启动ffplay
        threading.Thread(target=self.run_ffplay, args=(stream_url,), daemon=True).start()
    
    def run_ffplay(self, url):
        try:
            self.ffplay_process = subprocess.Popen(
                ["ffplay", "-i", url, "-fflags", "nobuffer", "-flags", "low_delay", "-framedrop"],
                stderr=subprocess.DEVNULL
            )
            self.ffplay_process.wait()
        except Exception as e:
            messagebox.showerror("错误", f"播放失败: {str(e)}")
        finally:
            self.reset_buttons()
    
    def stop_playback(self):
        if self.ffplay_process:
            self.ffplay_process.terminate()
            self.ffplay_process = None
        self.reset_buttons()
    
    def reset_buttons(self):
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def on_closing(self):
        self.stop_playback()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStreamPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()