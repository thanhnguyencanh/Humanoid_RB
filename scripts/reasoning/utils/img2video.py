import cv2
import os
import glob
import re
# import argparse

def numerical_sort_key(s):
    """
    Helper function to sort filenames numerically (natural sort).
    Extracts numbers from strings so 'frame_2.jpg' comes before 'frame_10.jpg'.
    """
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else s

def create_video_from_frames(input_folder, output_file, fps=30, ext="jpg"):
    """
    Combines images from a folder into a video file.
    """
    search_pattern = os.path.join(input_folder, f"*.{ext}")
    images = glob.glob(search_pattern)
    if not images:
        print(f"Error: No images found in {input_folder} with extension .{ext}")
        return
    # try:
    #     images.sort(key=numerical_sort_key)
    # except:
    #     print("Warning: Could not sort numerically. Falling back to standard sort.")
    #     images.sort()

    print(f"Found {len(images)} frames. Processing...")
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame.shape[:2] != (height, width):
            print(f"Skipping {img_path}: Size mismatch {frame.shape[:2]} vs {(height, width)}")
            continue

        out.write(frame)

        if idx % 50 == 0:
            print(f"Writing frame {idx}/{len(images)}...", end='\r')

    out.release()
    print(f"\nDone! Video saved as: {output_file}")


if __name__ == "__main__":
    input_dir = "/home/robot/thang_project/unitree_isaac/img2video"
    output_vid = "/home/robot/thang_project/unitree_isaac/scripts/reasoning/examples/videos/worker_video.mp4"
    if os.path.exists(input_dir):
        create_video_from_frames(input_dir, output_vid, fps=10, ext="png")
    else:
        print(f"Folder {input_dir} not found.")