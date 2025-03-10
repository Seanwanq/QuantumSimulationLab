import os
import re
from moviepy import ImageSequenceClip, concatenate_videoclips
from tqdm import tqdm


def animate_wigner(foldername, filename="wigner", fps=5, batch_size=10):
    png_files = [f for f in os.listdir(foldername) if f.endswith(".png")]

    def extract_time(name):
        match = re.search(r"wigner_time_(\d+(\.\d+)?).png", name)
        return float(match.group(1)) if match else float("inf")

    png_files.sort(key=extract_time)

    temp_clips = []
    for i in tqdm(range(0, len(png_files), batch_size)):
        batch = png_files[i : i + batch_size]
        clip = ImageSequenceClip([os.path.join(foldername, f) for f in batch], fps=fps)
        temp_clips.append(clip)

    final_clip = concatenate_videoclips(temp_clips)
    final_clip.write_videofile(filename + ".mp4", codec="libx264", fps=fps)
