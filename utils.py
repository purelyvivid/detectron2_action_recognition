from moviepy.editor import VideoFileClip

def display_video(VIDEO_FILE_PTH="out.avi"):
    clip=VideoFileClip(VIDEO_FILE_PTH)
    return clip.ipython_display(width=280)

