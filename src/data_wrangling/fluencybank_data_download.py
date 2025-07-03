import webbrowser
import time
import os
from pathlib import Path
from tqdm import tqdm


def open_links(links_file_path: str, delay_seconds: int = 5):
    """
    Opens each link from the file in the default browser with a delay between each link.

    Args:
        links_file_path (str): Path to the file containing video links
        delay_seconds (int): Delay in seconds between opening each link
    """
    # Check if file exists
    if not os.path.exists(links_file_path):
        print(f"Error: File {links_file_path} not found!")
        return

    # Read the links from the file
    with open(links_file_path, "r") as file:
        links = file.readlines()

    # Filter out empty lines and strip whitespace
    links = [link.strip() for link in links if link.strip()]

    print(f"Found {len(links)} links to open.")
    print(f"Opening each link with a {delay_seconds}-second delay...")
    print("Press Ctrl+C to stop at any time.\n")

    try:
        # Create progress bar
        with tqdm(total=len(links), desc="Opening links", unit="link") as pbar:
            for i, link in enumerate(links):
                pbar.set_description(f"Opening: {os.path.basename(link)}")
                webbrowser.open(link)
                pbar.update(1)

                # Don't wait after the last link
                if i < len(links) - 1:
                    time.sleep(delay_seconds)

        print("\nAll links have been opened!")

    except KeyboardInterrupt:
        print("\nStopped by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Path to the video links file
    video_links_file = "data/fluencybank/raw/Video/video_links.txt"
    cha_links_file = "data/fluencybank/raw/Chat/chat_links.txt"

    # Open the cha links with 5-second delays
    open_links(cha_links_file, delay_seconds=1)

    # Open the video links with 5-second delays
    # open_links(links_file, delay_seconds=100)
