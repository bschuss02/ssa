import webbrowser
import time
import os
from pathlib import Path
from tqdm import tqdm
import platform


def get_default_download_dir():
    """Get the default download directory for the current OS"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return os.path.expanduser("~/Downloads")
    elif system == "Windows":
        return os.path.expanduser("~/Downloads")
    else:  # Linux
        return os.path.expanduser("~/Downloads")


def get_active_downloads(download_dir):
    """
    Get currently active downloads by looking for temporary files and recently modified files
    """
    active_downloads = []
    download_path = Path(download_dir)

    if not download_path.exists():
        return active_downloads

    current_time = time.time()

    for file_path in download_path.iterdir():
        if file_path.is_file():
            # Check for temporary download files (common patterns)
            if file_path.suffix in [".part", ".tmp", ".crdownload"] or file_path.name.endswith(".download"):
                active_downloads.append(str(file_path))

            # Check for recently modified files (within last 10 seconds)
            elif current_time - file_path.stat().st_mtime < 10:
                # Additional check: see if file size is changing
                initial_size = file_path.stat().st_size
                time.sleep(0.5)
                try:
                    current_size = file_path.stat().st_size
                    if current_size != initial_size:
                        active_downloads.append(str(file_path))
                except (FileNotFoundError, OSError):
                    # File might have been moved/completed
                    pass

    return active_downloads


def wait_for_download_slot(download_dir, max_concurrent=2, check_interval=2):
    """
    Wait until there's a free download slot (less than max_concurrent downloads)
    """
    while True:
        active = get_active_downloads(download_dir)
        if len(active) < max_concurrent:
            return True

        print(f"  Waiting for download slot... ({len(active)}/{max_concurrent} active)")
        time.sleep(check_interval)


def wait_for_download_completion(download_dir, filename, timeout=300):
    """
    Wait for a specific download to complete by monitoring file stability
    """
    file_path = Path(download_dir) / filename
    start_time = time.time()

    # Wait for file to appear
    while not file_path.exists() and time.time() - start_time < timeout:
        time.sleep(1)

    if not file_path.exists():
        return False

    # Wait for file size to stabilize
    stable_count = 0
    last_size = 0

    while stable_count < 3 and time.time() - start_time < timeout:
        try:
            current_size = file_path.stat().st_size
            if current_size == last_size and current_size > 0:
                stable_count += 1
            else:
                stable_count = 0
            last_size = current_size
            time.sleep(1)
        except (FileNotFoundError, OSError):
            # File might have been moved/renamed
            break

    return stable_count >= 3


def extract_filename_from_url(url):
    """Extract expected filename from URL"""
    return url.split("/")[-1].split("?")[0]


def open_links_with_download_management(
    links_file_path: str, delay_seconds: int = 1, max_concurrent: int = 2
):
    """
    Opens each link from the file in the default browser with download management.

    Args:
        links_file_path (str): Path to the file containing links
        delay_seconds (int): Delay in seconds between opening each link
        max_concurrent (int): Maximum number of concurrent downloads
    """
    # Check if file exists
    if not os.path.exists(links_file_path):
        print(f"Error: File {links_file_path} not found!")
        return

    # Get download directory
    download_dir = get_default_download_dir()
    print(f"Monitoring downloads in: {download_dir}")

    # Read the links from the file
    with open(links_file_path, "r") as file:
        links = file.readlines()

    # Filter out empty lines and strip whitespace
    links = [link.strip() for link in links if link.strip()]

    print(f"Found {len(links)} links to open.")
    print(f"Maximum concurrent downloads: {max_concurrent}")
    print(f"Delay between links: {delay_seconds} seconds")
    print("Press Ctrl+C to stop at any time.\n")

    try:
        # Create progress bar
        with tqdm(total=len(links), desc="Processing links", unit="link") as pbar:
            for i, link in enumerate(links):
                filename = extract_filename_from_url(link)
                pbar.set_description(f"Opening: {filename}")

                # Wait for available download slot
                wait_for_download_slot(download_dir, max_concurrent)

                # Open the link
                webbrowser.open(link)
                pbar.update(1)

                # Small delay to allow browser to start download
                time.sleep(delay_seconds)

                # Show current status
                active = get_active_downloads(download_dir)
                if active:
                    active_names = [os.path.basename(f) for f in active]
                    print(
                        f"  Active downloads ({len(active)}): {', '.join(active_names[:3])}{'...' if len(active_names) > 3 else ''}"
                    )

        print(f"\nAll {len(links)} links have been opened!")

        # Wait for all downloads to complete
        print("Waiting for all downloads to complete...")
        final_pbar = tqdm(desc="Waiting for downloads to finish", unit="check")

        while True:
            active = get_active_downloads(download_dir)
            if not active:
                break

            final_pbar.set_description(f"Waiting for {len(active)} downloads to finish")
            final_pbar.update(1)
            time.sleep(5)

        final_pbar.close()
        print("All downloads completed!")

    except KeyboardInterrupt:
        print("\nStopped by user. Exiting...")
        active = get_active_downloads(download_dir)
        if active:
            print(f"Note: {len(active)} downloads may still be in progress")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Path to the links files
    video_links_file = "data/fluencybank/raw/video_links.txt"
    cha_links_file = "data/fluencybank/raw/chat_links.txt"

    # Open the video links with managed concurrent downloads
    open_links_with_download_management(video_links_file, delay_seconds=1, max_concurrent=3)

    # Uncomment to also process chat links
    # open_links_with_download_management(cha_links_file, delay_seconds=1, max_concurrent=2)
