#!/bin/bash

# Script to load videos for image detection and tracking

# Define variables
VIDEO_DIR="videos"
OUTPUT_DIR="outputs"
VIDEO_LIST="list.txt"
LIVE_STREAM_LIST="live_stream_list.txt"

# Function to check if directories exist, create if not
check_directories() {
    if [ ! -d "$VIDEO_DIR" ]; then
        echo "Video directory does not exist. Creating: $VIDEO_DIR"
        mkdir -p "$VIDEO_DIR"
    fi

    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Output directory does not exist. Creating: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
}

# Function to download videos using yt-dlp
download_videos() {
    while IFS= read -r url; do
        echo "Downloading video from URL: $url"
        yt-dlp -o "$VIDEO_DIR/%(title)s.%(ext)s" "$url"
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded: $url"
        else
            echo "Error downloading: $url" >&2
        fi
    done < "$VIDEO_LIST"
}

# Function to download live stream videos using yt-dlp
download_live_streams() {
    while IFS= read -r url; do
        echo "Downloading live stream from URL: $url"
        yt-dlp --live-from-start --max-downloads 1 --external-downloader ffmpeg --external-downloader-args "-t 00:15:00" -o "$VIDEO_DIR/%(title)s.%(ext)s" "$url"
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded live stream: $url"
        else
            echo "Error downloading live stream: $url" >&2
        fi
    done < "$LIVE_STREAM_LIST"
}

# Main script execution
check_directories
download_videos
download_live_streams
