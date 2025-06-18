import yt_dlp

# URL of the YouTube video to download
VIDEO_URL = "https://youtu.be/DYxbtpECZAY"

# yt-dlp options: download as mp4 and save as gait_video.mp4
ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'gait_video.mp4'
}

# Download the video using yt-dlp
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([VIDEO_URL])
