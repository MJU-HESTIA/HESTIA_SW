import tkinter as tk
from moviepy.editor import VideoFileClip

# GUI 설계
win = tk.Tk()

# 풀스크린 모드 설정, 키 설정 (F11 전체화면, Escape 전체화면탈출)
win.attributes("-fullscreen", True)
win.bind("<F11>", lambda event: win.attributes("-fullscreen", not win.attributes("-fullscreen")))
win.bind("<Escape>", lambda event: win.attributes("-fullscreen", False))

# 비디오 파일 경로
video_path = "intro.mp4"
# moviepy를 사용해 비디오 로드
video_clip = VideoFileClip(video_path)

# 비디오 재생 및 오디오 출력
video_clip.preview(fps=30, audio=True)

# 영상이 끝난 후 윈도우 닫기
win.destroy()