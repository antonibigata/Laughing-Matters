import os
import cv2
import argparse
from tqdm import tqdm


def convert_videos_to_images(videos_folder, output_folder):
    # Iterate over each file in the videos folder
    for root, _, files in os.walk(videos_folder):
        for filename in files:
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                # Get the name of the video (without extension)
                video_name = os.path.splitext(filename)[0]

                # Get the relative path of the video folder
                relative_folder = os.path.relpath(root, videos_folder)

                # Create the output folder path
                output_video_folder = os.path.join(output_folder, relative_folder)
                os.makedirs(output_video_folder, exist_ok=True)

                # Create the output video file path
                output_video_path = os.path.join(output_video_folder, video_name)

                # Read the video file
                video_path = os.path.join(root, filename)
                cap = cv2.VideoCapture(video_path)

                # Read and save each frame as an image
                frame_count = 0
                pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Save the frame as an image
                    image_path = f"{output_video_path}_{frame_count:04d}.jpg"
                    cv2.imwrite(image_path, frame)

                    frame_count += 1
                    pbar.update(1)

                # Release the video capture object
                cap.release()
                pbar.close()


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert videos to images")
parser.add_argument("videos_folder", help="Path to the folder containing the videos")
parser.add_argument("output_folder", help="Path to the folder where images will be saved")
args = parser.parse_args()

# Call the function to convert videos to images
convert_videos_to_images(args.videos_folder, args.output_folder)
