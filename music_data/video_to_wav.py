import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# Define the directories
VIDEO_DIR = "music_data/data/kpop"
MUSIC_WAV_DIR = "music_data/music_wav"

# Create the output directory if it doesn't exist
os.makedirs(MUSIC_WAV_DIR, exist_ok=True)

# Iterate through files in the video directory
for filename in os.listdir(VIDEO_DIR):
    if filename.lower().endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, filename)
        # Construct the output WAV filename
        base_filename = os.path.splitext(filename)[0]
        wav_filename = f"{base_filename}.wav"
        wav_path = os.path.join(MUSIC_WAV_DIR, wav_filename)

        print(f"Converting {filename} to {wav_filename}...")

        try:
            # Load the video file
            video_clip = VideoFileClip(video_path)
            # Extract the audio
            audio_clip = video_clip.audio
            # Write the audio to a WAV file
            # Use codec='pcm_s16le' for standard WAV format
            audio_clip.write_audiofile(wav_path, codec="pcm_s16le")
            # Close the clips to release resources
            audio_clip.close()
            video_clip.close()
            print(f"Successfully converted {filename} to {wav_filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("Conversion process finished.")
