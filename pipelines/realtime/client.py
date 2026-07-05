

## chunk and send video to modal endpoint

import asyncio
import os
import websockets
import re
import requests
from pathlib import Path
import cv2
from tqdm import tqdm
from time import sleep

URI = "wss://sopleee--websocket-ex-sample-endpt.modal.run/ws"
VID_LINK = "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
VID_PATH = "./artifacts/tempvid.mp4"

async def send_data(): 
    async with websockets.connect(URI) as websocket: 
        message = "hi from client"
        print(f"Sending {message}")
        await websocket.send(message)
        response = await websocket.recv()
        print(f"Received: {response}")

def camera_mock_frame_gen(vid_path):
    camera_mock = cv2.VideoCapture(vid_path)
    video_fps = camera_mock.get(cv2.CAP_PROP_FPS)
    total_frames = int(camera_mock.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Inference Progress")
    sleep_amt = 1/video_fps # sleep per frame sent to match fps
    
    while True: 
        ret, frame = camera_mock.read()
        if not ret: break
        
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret: continue
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
        sleep(sleep_amt)

async def send_vid_frame_by_fps(camera_stream): 
    async with websockets.connect(URI) as websocket:
        frame_i = 0
        for f in camera_stream:
            await websocket.send(f)
            print(f"Sent: {frame_i}")
            response = await websocket.recv()
            print(f"Received: {response}")
            frame_i+=1

def extract_file_id(share_link):
    """Extract file ID from various Google Drive link formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, share_link)
        if match: return match.group(1)
    
    return None

def confirm_download_res(path):
    with open(path, 'rb') as f:
        first_bytes = f.read(100)
        # Check if file is HTML (common issue with Google Drive)
        if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
            print("ERROR: Downloaded file is HTML, not a video!")
            print("The Google Drive link may not be publicly accessible.")
            print(f"First 100 bytes: {first_bytes[:100]}")
            return False
        # Check for valid video file signature
        if not (first_bytes.startswith(b'\x00\x00\x00') or  # MP4/MOV
                first_bytes.startswith(b'ftyp') or
                b'ftyp' in first_bytes[:20]):
            print(f"WARNING: File may not be a valid video")
            print(f"First 20 bytes: {first_bytes[:20]}")
    print(f"\n✓ Downloaded successfully to: {path}")
    return True

def download_vid(link, path):
    """
    Download a file from a public Google Drive link.
    
    Args:
        share_link: Google Drive share link (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
        output_path: Path where the file will be saved (e.g., 'video.mp4')
    """
    file_id = extract_file_id(link)
    if not file_id:
        print("Error: Could not extract file ID from the link")
        return False
    
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    try: response = session.get(url, stream=True)
    except Exception as e:
        print(f"\nNetwork related error: {e}")
        return False
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if not token:
        for line in response.iter_lines():
            if b'confirm=' in line:
                match = re.search(b'confirm=([^&"]+)', line)
                if match:
                    token = match.group(1).decode()
                    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
                    break
        response = session.get(url, stream=True)
    
    if token:
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
        response = session.get(url, stream=True)
    elif not any(key.startswith('download_warning') for key in response.cookies.keys()):
        response = session.get(url, stream=True)
          
    if 'text/html' in response.headers.get('Content-Type', ''):
        # This is the virus warning page for large files
        html_content = response.text
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)
        
        if uuid_match:
            uuid = uuid_match.group(1)
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={uuid}"
            print(f"Large file detected, using confirmation download...")
        else:
            # Fallback: try without UUID
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        response = session.get(url, stream=True)

    # Download the file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        
        confirm_download_res(path)
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False    
    
if __name__ == "__main__": 
    download_vid(VID_LINK, VID_PATH)
    # asyncio.run(send_data())
    asyncio.run(send_vid_frame_by_fps(camera_mock_frame_gen(VID_PATH)))
    os.remove(VID_PATH)


# input = local video path (good enough?)
# pass in messages frame by frame
# keep track of the rate of messages received per second
# on the server end, keep track of the rate of messages received per second

# do the yolo processing stuff (just a single image at a time to start)
# have it write to some sort of python canvas on the client