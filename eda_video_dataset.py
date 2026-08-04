#!/usr/bin/env python3
"""
Comprehensive EDA for HCMAI25 Video Dataset
============================================
Analyzes 605 videos across 20 groups (K01-K20) to guide preprocessing 
for 4 tasks: KIS, AVS, VQA, KISC.

Outputs:
  - eda_output/video_metadata.csv          : Full metadata table
  - eda_output/01_duration_distribution.png : Duration histogram + boxplot
  - eda_output/02_resolution_fps.png        : Resolution & FPS analysis
  - eda_output/03_filesize_codec.png        : File size & codec breakdown
  - eda_output/04_frames_analysis.png       : Frame count analysis
  - eda_output/05_group_statistics.png      : Per-group statistics
  - eda_output/06_brightness_contrast.png   : Visual quality sampling
  - eda_output/07_correlation_matrix.png    : Feature correlations
  - eda_output/08_temporal_density.png      : Temporal density analysis
  - eda_output/eda_report.txt              : Text summary report
"""

import os
import sys
import json
import subprocess
import csv
import math
import random
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Configuration ──────────────────────────────────────────────────────────
VIDEO_DIR = "/mlcv2025/Datasets/HCMAI25/batch2/video"
OUTPUT_DIR = "/workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026/eda_output"
SAMPLE_FRAMES_PER_VIDEO = 5   # Number of frames to sample per video for visual analysis
MAX_WORKERS = 8               # Parallel ffprobe workers
BRIGHTNESS_SAMPLE_VIDEOS = 60 # How many videos to sample for brightness/contrast

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style Configuration ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

COLORS = ['#e94560', '#0f3460', '#533483', '#16c79a', '#f7b731',
          '#eb3b5a', '#4b7bec', '#a55eea', '#26de81', '#fd9644']

GRADIENT_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'custom', ['#0f3460', '#e94560', '#f7b731'], N=256)


def get_video_metadata(video_path):
    """Extract metadata from a single video using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        for s in data.get('streams', []):
            if s.get('codec_type') == 'video' and video_stream is None:
                video_stream = s
            elif s.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = s

        if video_stream is None:
            return None

        fmt = data.get('format', {})
        
        # Parse FPS
        fps_str = video_stream.get('r_frame_rate', '0/1')
        try:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        except (ValueError, ZeroDivisionError):
            fps = 0

        filename = os.path.basename(video_path)
        group = filename.split('_')[0]  # e.g., K01
        
        metadata = {
            'filename': filename,
            'group': group,
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': round(fps, 2),
            'duration_sec': float(video_stream.get('duration', fmt.get('duration', 0))),
            'codec': video_stream.get('codec_name', 'unknown'),
            'nb_frames': int(video_stream.get('nb_frames', 0)),
            'file_size_mb': round(int(fmt.get('size', 0)) / (1024 * 1024), 2),
            'bitrate_kbps': round(int(fmt.get('bit_rate', 0)) / 1000, 2),
            'has_audio': audio_stream is not None,
            'audio_codec': audio_stream.get('codec_name', 'none') if audio_stream else 'none',
            'audio_sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
            'audio_channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'resolution': f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
        }
        
        # Compute derived metrics
        metadata['duration_min'] = round(metadata['duration_sec'] / 60, 2)
        metadata['aspect_ratio'] = round(metadata['width'] / metadata['height'], 4) if metadata['height'] > 0 else 0
        metadata['total_pixels'] = metadata['width'] * metadata['height']
        metadata['bitrate_per_pixel'] = round(
            (metadata['bitrate_kbps'] * 1000) / (metadata['total_pixels'] * metadata['fps'])
            if metadata['total_pixels'] > 0 and metadata['fps'] > 0 else 0, 6)
        
        return metadata
    except Exception as e:
        print(f"  Error processing {video_path}: {e}")
        return None


def collect_all_metadata(video_dir):
    """Collect metadata from all videos in parallel."""
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_video_metadata, vf): vf for vf in video_files}
        for i, future in enumerate(as_completed(futures)):
            meta = future.result()
            if meta:
                results.append(meta)
            if (i + 1) % 50 == 0 or i + 1 == len(video_files):
                print(f"  Processed {i+1}/{len(video_files)} videos...")
    
    results.sort(key=lambda x: x['filename'])
    return results


def sample_brightness_contrast(video_dir, metadata_list, n_samples, frames_per_video):
    """Sample frames from a subset of videos to measure brightness and contrast."""
    print(f"\nSampling brightness/contrast from {n_samples} videos...")
    
    sampled = random.sample(metadata_list, min(n_samples, len(metadata_list)))
    brightness_data = []
    
    for i, meta in enumerate(sampled):
        video_path = os.path.join(video_dir, meta['filename'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
        
        # Sample evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
        
        video_brightness = []
        video_contrast = []
        video_saturation = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            video_brightness.append(float(np.mean(gray)))
            video_contrast.append(float(np.std(gray)))
            video_saturation.append(float(np.mean(hsv[:, :, 1])))
        
        cap.release()
        
        if video_brightness:
            brightness_data.append({
                'filename': meta['filename'],
                'group': meta['group'],
                'mean_brightness': np.mean(video_brightness),
                'std_brightness': np.std(video_brightness),
                'mean_contrast': np.mean(video_contrast),
                'std_contrast': np.std(video_contrast),
                'mean_saturation': np.mean(video_saturation),
                'std_saturation': np.std(video_saturation),
                'min_brightness': min(video_brightness),
                'max_brightness': max(video_brightness),
            })
        
        if (i + 1) % 10 == 0:
            print(f"  Sampled {i+1}/{len(sampled)} videos...")
    
    return brightness_data


def save_metadata_csv(metadata_list, output_path):
    """Save metadata to CSV."""
    if not metadata_list:
        return
    fieldnames = list(metadata_list[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    print(f"Saved metadata to {output_path}")


# ── Visualization Functions ───────────────────────────────────────────────

def plot_duration_distribution(metadata_list, output_path):
    """Plot 1: Duration distribution."""
    durations_min = [m['duration_min'] for m in metadata_list]
    durations_sec = [m['duration_sec'] for m in metadata_list]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('📊 Video Duration Analysis', fontsize=18, fontweight='bold', color='#e94560')
    
    # Histogram of durations in minutes
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(durations_min, bins=30, color='#e94560', alpha=0.8, edgecolor='#333')
    # Color gradient
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GRADIENT_CMAP(b / max(bins)))
    ax1.set_xlabel('Duration (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title('Duration Distribution')
    ax1.axvline(np.mean(durations_min), color='#f7b731', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(durations_min):.1f} min')
    ax1.axvline(np.median(durations_min), color='#16c79a', linestyle='--', linewidth=2,
                label=f'Median: {np.median(durations_min):.1f} min')
    ax1.legend(fontsize=9)
    
    # Boxplot per group
    ax2 = fig.add_subplot(gs[0, 1])
    groups = sorted(set(m['group'] for m in metadata_list))
    group_durations = [[m['duration_min'] for m in metadata_list if m['group'] == g] for g in groups]
    bp = ax2.boxplot(group_durations, labels=groups, patch_artist=True, 
                     medianprops=dict(color='#f7b731', linewidth=2))
    for patch, color in zip(bp['boxes'], plt.cm.coolwarm(np.linspace(0, 1, len(groups)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel('Video Group')
    ax2.set_ylabel('Duration (minutes)')
    ax2.set_title('Duration by Group')
    ax2.tick_params(axis='x', rotation=45)
    
    # CDF of durations
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_dur = np.sort(durations_min)
    cdf = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
    ax3.plot(sorted_dur, cdf, color='#e94560', linewidth=2)
    ax3.fill_between(sorted_dur, cdf, alpha=0.2, color='#e94560')
    ax3.set_xlabel('Duration (minutes)')
    ax3.set_ylabel('Cumulative Proportion')
    ax3.set_title('Cumulative Distribution of Duration')
    # Add percentile markers
    for pct in [25, 50, 75, 90]:
        val = np.percentile(durations_min, pct)
        ax3.axhline(pct/100, color='#555', linestyle=':', alpha=0.5)
        ax3.axvline(val, color='#16c79a', linestyle=':', alpha=0.5)
        ax3.annotate(f'P{pct}: {val:.1f}m', xy=(val, pct/100), fontsize=8,
                     color='#16c79a', ha='left')
    
    # Statistics text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    stats_text = (
        f"📈 Duration Statistics\n"
        f"{'─' * 35}\n"
        f"Total Videos:    {len(durations_min)}\n"
        f"Total Duration:  {sum(durations_min):.1f} min ({sum(durations_min)/60:.1f} hrs)\n"
        f"Mean Duration:   {np.mean(durations_min):.2f} min\n"
        f"Median Duration: {np.median(durations_min):.2f} min\n"
        f"Std Dev:         {np.std(durations_min):.2f} min\n"
        f"Min Duration:    {min(durations_min):.2f} min\n"
        f"Max Duration:    {max(durations_min):.2f} min\n"
        f"{'─' * 35}\n"
        f"P25: {np.percentile(durations_min, 25):.2f} min\n"
        f"P75: {np.percentile(durations_min, 75):.2f} min\n"
        f"P90: {np.percentile(durations_min, 90):.2f} min\n"
        f"P95: {np.percentile(durations_min, 95):.2f} min\n"
        f"IQR: {np.percentile(durations_min, 75) - np.percentile(durations_min, 25):.2f} min"
    )
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#0f3460', alpha=0.8, edgecolor='#e94560'))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_resolution_fps(metadata_list, output_path):
    """Plot 2: Resolution and FPS analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('🖥️ Resolution & FPS Analysis', fontsize=18, fontweight='bold', color='#e94560')
    
    # Resolution distribution
    ax1 = fig.add_subplot(gs[0, 0])
    resolutions = Counter(m['resolution'] for m in metadata_list)
    res_labels = list(resolutions.keys())
    res_counts = list(resolutions.values())
    bars = ax1.barh(res_labels, res_counts, color=COLORS[:len(res_labels)], alpha=0.85)
    for bar, count in zip(bars, res_counts):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{count}', va='center', fontsize=10, color='#eee')
    ax1.set_xlabel('Count')
    ax1.set_title('Resolution Distribution')
    
    # FPS distribution
    ax2 = fig.add_subplot(gs[0, 1])
    fps_values = [m['fps'] for m in metadata_list]
    fps_counter = Counter(fps_values)
    fps_labels = [str(k) for k in sorted(fps_counter.keys())]
    fps_counts = [fps_counter[k] for k in sorted(fps_counter.keys())]
    bars = ax2.bar(fps_labels, fps_counts, color='#533483', alpha=0.85, edgecolor='#e94560')
    for bar, count in zip(bars, fps_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{count}', ha='center', fontsize=10, color='#eee')
    ax2.set_xlabel('FPS')
    ax2.set_ylabel('Count')
    ax2.set_title('Frame Rate Distribution')
    
    # Aspect ratio distribution
    ax3 = fig.add_subplot(gs[1, 0])
    aspect_ratios = [m['aspect_ratio'] for m in metadata_list]
    ar_counter = Counter([round(ar, 2) for ar in aspect_ratios])
    ar_labels = [str(k) for k in sorted(ar_counter.keys())]
    ar_counts = [ar_counter[k] for k in sorted(ar_counter.keys())]
    bars = ax3.bar(ar_labels, ar_counts, color='#16c79a', alpha=0.85, edgecolor='#333')
    for bar, count in zip(bars, ar_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{count}', ha='center', fontsize=9, color='#eee')
    ax3.set_xlabel('Aspect Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Aspect Ratio Distribution')
    
    # Pixel format distribution
    ax4 = fig.add_subplot(gs[1, 1])
    pix_fmts = Counter(m['pixel_format'] for m in metadata_list)
    pf_labels = list(pix_fmts.keys())
    pf_counts = list(pix_fmts.values())
    wedges, texts, autotexts = ax4.pie(pf_counts, labels=pf_labels, autopct='%1.1f%%',
                                        colors=COLORS[:len(pf_labels)],
                                        textprops={'color': '#eee', 'fontsize': 10})
    ax4.set_title('Pixel Format Distribution')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_filesize_codec(metadata_list, output_path):
    """Plot 3: File size and codec analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('💾 File Size & Codec Analysis', fontsize=18, fontweight='bold', color='#e94560')
    
    # File size histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [m['file_size_mb'] for m in metadata_list]
    n, bins, patches = ax1.hist(sizes, bins=30, alpha=0.8, edgecolor='#333')
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GRADIENT_CMAP(b / max(bins)))
    ax1.set_xlabel('File Size (MB)')
    ax1.set_ylabel('Count')
    ax1.set_title('File Size Distribution')
    ax1.axvline(np.mean(sizes), color='#f7b731', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(sizes):.0f} MB')
    ax1.legend()
    
    # Codec distribution
    ax2 = fig.add_subplot(gs[0, 1])
    codecs = Counter(m['codec'] for m in metadata_list)
    codec_labels = list(codecs.keys())
    codec_counts = list(codecs.values())
    bars = ax2.bar(codec_labels, codec_counts, color=['#e94560', '#0f3460', '#533483', '#16c79a'][:len(codec_labels)],
                   alpha=0.85, edgecolor='#333')
    for bar, count in zip(bars, codec_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{count}', ha='center', fontsize=11, color='#eee')
    ax2.set_xlabel('Codec')
    ax2.set_ylabel('Count')
    ax2.set_title('Video Codec Distribution')
    
    # File size per codec (boxplot)
    ax3 = fig.add_subplot(gs[1, 0])
    codec_sizes = defaultdict(list)
    for m in metadata_list:
        codec_sizes[m['codec']].append(m['file_size_mb'])
    codec_keys = sorted(codec_sizes.keys())
    bp = ax3.boxplot([codec_sizes[k] for k in codec_keys], labels=codec_keys,
                     patch_artist=True, medianprops=dict(color='#f7b731', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xlabel('Codec')
    ax3.set_ylabel('File Size (MB)')
    ax3.set_title('File Size by Codec')
    
    # Bitrate distribution
    ax4 = fig.add_subplot(gs[1, 1])
    bitrates = [m['bitrate_kbps'] for m in metadata_list if m['bitrate_kbps'] > 0]
    n, bins, patches = ax4.hist(bitrates, bins=30, alpha=0.8, edgecolor='#333')
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GRADIENT_CMAP(b / max(bins)))
    ax4.set_xlabel('Bitrate (kbps)')
    ax4.set_ylabel('Count')
    ax4.set_title('Bitrate Distribution')
    ax4.axvline(np.mean(bitrates), color='#f7b731', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(bitrates):.0f} kbps')
    ax4.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_frames_analysis(metadata_list, output_path):
    """Plot 4: Frame count analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('🎬 Frame Count Analysis', fontsize=18, fontweight='bold', color='#e94560')
    
    frames = [m['nb_frames'] for m in metadata_list if m['nb_frames'] > 0]
    
    # Frame count histogram
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(frames, bins=30, alpha=0.8, edgecolor='#333')
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GRADIENT_CMAP(b / max(bins)))
    ax1.set_xlabel('Frame Count')
    ax1.set_ylabel('Number of Videos')
    ax1.set_title('Frame Count Distribution')
    ax1.axvline(np.mean(frames), color='#f7b731', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(frames):.0f}')
    ax1.legend()
    
    # Frames vs Duration scatter
    ax2 = fig.add_subplot(gs[0, 1])
    durations = [m['duration_sec'] for m in metadata_list if m['nb_frames'] > 0]
    fps_vals = [m['fps'] for m in metadata_list if m['nb_frames'] > 0]
    scatter = ax2.scatter(durations, frames, c=fps_vals, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax2, label='FPS')
    ax2.set_xlabel('Duration (sec)')
    ax2.set_ylabel('Frame Count')
    ax2.set_title('Frames vs Duration (colored by FPS)')
    
    # Frames per group
    ax3 = fig.add_subplot(gs[1, 0])
    groups = sorted(set(m['group'] for m in metadata_list))
    group_frames = [np.mean([m['nb_frames'] for m in metadata_list if m['group'] == g and m['nb_frames'] > 0]) 
                    for g in groups]
    bars = ax3.bar(groups, group_frames, color=[GRADIENT_CMAP(i/len(groups)) for i in range(len(groups))],
                   alpha=0.85, edgecolor='#333')
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Avg Frame Count')
    ax3.set_title('Average Frame Count per Group')
    ax3.tick_params(axis='x', rotation=45)
    
    # Total frames summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    total_frames = sum(frames)
    stats_text = (
        f"🎬 Frame Count Statistics\n"
        f"{'─' * 35}\n"
        f"Total Frames:    {total_frames:,}\n"
        f"Mean Frames:     {np.mean(frames):,.0f}\n"
        f"Median Frames:   {np.median(frames):,.0f}\n"
        f"Std Dev:         {np.std(frames):,.0f}\n"
        f"Min Frames:      {min(frames):,}\n"
        f"Max Frames:      {max(frames):,}\n"
        f"{'─' * 35}\n"
        f"Keyframe Extraction Estimates:\n"
        f"  @1 fps:  {sum(m['duration_sec'] for m in metadata_list):,.0f} frames\n"
        f"  @2 fps:  {sum(m['duration_sec'] for m in metadata_list)*2:,.0f} frames\n"
        f"  @0.5fps: {sum(m['duration_sec'] for m in metadata_list)/2:,.0f} frames\n"
        f"{'─' * 35}\n"
        f"Disk space for keyframes (est):\n"
        f"  @1fps, 100KB/f: {sum(m['duration_sec'] for m in metadata_list)*100/1024/1024:.1f} GB\n"
        f"  @1fps, 50KB/f:  {sum(m['duration_sec'] for m in metadata_list)*50/1024/1024:.1f} GB"
    )
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#0f3460', alpha=0.8, edgecolor='#e94560'))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_group_statistics(metadata_list, output_path):
    """Plot 5: Per-group comprehensive statistics."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('📁 Per-Group Statistics', fontsize=18, fontweight='bold', color='#e94560')
    
    groups = sorted(set(m['group'] for m in metadata_list))
    
    # Videos per group
    ax1 = fig.add_subplot(gs[0, 0])
    group_counts = [sum(1 for m in metadata_list if m['group'] == g) for g in groups]
    bars = ax1.bar(groups, group_counts, color=[GRADIENT_CMAP(i/len(groups)) for i in range(len(groups))],
                   alpha=0.85, edgecolor='#333')
    for bar, count in zip(bars, group_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{count}', ha='center', fontsize=9, color='#eee')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Number of Videos')
    ax1.set_title('Videos per Group')
    ax1.tick_params(axis='x', rotation=45)
    
    # Total duration per group (hours)
    ax2 = fig.add_subplot(gs[0, 1])
    group_total_dur = [sum(m['duration_min'] for m in metadata_list if m['group'] == g) / 60 
                       for g in groups]
    bars = ax2.bar(groups, group_total_dur, color=[GRADIENT_CMAP(i/len(groups)) for i in range(len(groups))],
                   alpha=0.85, edgecolor='#333')
    for bar, dur in zip(bars, group_total_dur):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{dur:.1f}h', ha='center', fontsize=8, color='#eee')
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Total Duration (hours)')
    ax2.set_title('Total Duration per Group')
    ax2.tick_params(axis='x', rotation=45)
    
    # Total size per group (GB)
    ax3 = fig.add_subplot(gs[1, 0])
    group_total_size = [sum(m['file_size_mb'] for m in metadata_list if m['group'] == g) / 1024 
                        for g in groups]
    bars = ax3.bar(groups, group_total_size, color=[GRADIENT_CMAP(i/len(groups)) for i in range(len(groups))],
                   alpha=0.85, edgecolor='#333')
    for bar, size in zip(bars, group_total_size):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{size:.1f}G', ha='center', fontsize=8, color='#eee')
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Total Size (GB)')
    ax3.set_title('Total File Size per Group')
    ax3.tick_params(axis='x', rotation=45)
    
    # Codec distribution per group (stacked bar)
    ax4 = fig.add_subplot(gs[1, 1])
    all_codecs = sorted(set(m['codec'] for m in metadata_list))
    codec_colors = {'h264': '#e94560', 'av1': '#0f3460', 'hevc': '#533483', 'vp9': '#16c79a'}
    
    bottom = np.zeros(len(groups))
    for codec in all_codecs:
        counts = [sum(1 for m in metadata_list if m['group'] == g and m['codec'] == codec) for g in groups]
        color = codec_colors.get(codec, '#f7b731')
        ax4.bar(groups, counts, bottom=bottom, label=codec, color=color, alpha=0.85, edgecolor='#333')
        bottom += np.array(counts)
    ax4.set_xlabel('Group')
    ax4.set_ylabel('Count')
    ax4.set_title('Codec Distribution per Group')
    ax4.legend(fontsize=9)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_brightness_contrast(brightness_data, output_path):
    """Plot 6: Brightness and contrast analysis."""
    if not brightness_data:
        print("  No brightness data to plot. Skipping.")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('🔆 Visual Quality Analysis (Sampled)', fontsize=18, fontweight='bold', color='#e94560')
    
    brightness_vals = [d['mean_brightness'] for d in brightness_data]
    contrast_vals = [d['mean_contrast'] for d in brightness_data]
    saturation_vals = [d['mean_saturation'] for d in brightness_data]
    
    # Brightness distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(brightness_vals, bins=25, color='#f7b731', alpha=0.8, edgecolor='#333')
    ax1.axvline(np.mean(brightness_vals), color='#e94560', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(brightness_vals):.1f}')
    ax1.set_xlabel('Mean Brightness (0-255)')
    ax1.set_ylabel('Count')
    ax1.set_title('Brightness Distribution')
    ax1.legend()
    # Mark low/high brightness zones
    ax1.axvspan(0, 50, alpha=0.1, color='red', label='Dark')
    ax1.axvspan(200, 255, alpha=0.1, color='yellow', label='Bright')
    
    # Contrast distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(contrast_vals, bins=25, color='#533483', alpha=0.8, edgecolor='#333')
    ax2.axvline(np.mean(contrast_vals), color='#e94560', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(contrast_vals):.1f}')
    ax2.set_xlabel('Mean Contrast (Std Dev of Gray)')
    ax2.set_ylabel('Count')
    ax2.set_title('Contrast Distribution')
    ax2.legend()
    
    # Brightness vs Contrast scatter
    ax3 = fig.add_subplot(gs[1, 0])
    groups = list(set(d['group'] for d in brightness_data))
    for i, g in enumerate(sorted(groups)):
        gdata = [d for d in brightness_data if d['group'] == g]
        ax3.scatter([d['mean_brightness'] for d in gdata], 
                    [d['mean_contrast'] for d in gdata],
                    c=COLORS[i % len(COLORS)], alpha=0.6, s=30, label=g)
    ax3.set_xlabel('Mean Brightness')
    ax3.set_ylabel('Mean Contrast')
    ax3.set_title('Brightness vs Contrast')
    # Don't show legend if too many groups
    if len(groups) <= 10:
        ax3.legend(fontsize=7, ncol=2)
    
    # Saturation distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(saturation_vals, bins=25, color='#16c79a', alpha=0.8, edgecolor='#333')
    ax4.axvline(np.mean(saturation_vals), color='#e94560', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(saturation_vals):.1f}')
    ax4.set_xlabel('Mean Saturation (0-255)')
    ax4.set_ylabel('Count')
    ax4.set_title('Color Saturation Distribution')
    ax4.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correlation_matrix(metadata_list, output_path):
    """Plot 7: Feature correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('🔗 Feature Correlation Matrix', fontsize=18, fontweight='bold', color='#e94560')
    
    features = ['duration_sec', 'width', 'height', 'fps', 'nb_frames', 
                'file_size_mb', 'bitrate_kbps', 'aspect_ratio']
    
    data = np.array([[m[f] for f in features] for m in metadata_list], dtype=float)
    
    # Compute correlation matrix
    n = len(features)
    corr = np.corrcoef(data.T)
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(features, fontsize=9)
    
    # Add correlation values
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr[i, j]) > 0.5 else '#eee'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', 
                    fontsize=8, color=color, fontweight='bold')
    
    ax.set_title('Numerical Feature Correlations', pad=15)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_temporal_density(metadata_list, output_path):
    """Plot 8: Temporal density analysis - useful for keyframe extraction planning."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('⏱️ Temporal & Keyframe Extraction Planning', fontsize=18, fontweight='bold', color='#e94560')
    
    durations = [m['duration_sec'] for m in metadata_list]
    fps_vals = [m['fps'] for m in metadata_list]
    
    # Keyframe counts at different extraction rates
    ax1 = fig.add_subplot(gs[0, 0])
    extraction_rates = [0.25, 0.5, 1, 2, 3, 5]
    total_keyframes = [sum(d * r for d in durations) for r in extraction_rates]
    bars = ax1.bar([f'{r} fps' for r in extraction_rates], total_keyframes,
                   color=[GRADIENT_CMAP(i/len(extraction_rates)) for i in range(len(extraction_rates))],
                   alpha=0.85, edgecolor='#333')
    for bar, count in zip(bars, total_keyframes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'{count/1000:.0f}K', ha='center', fontsize=9, color='#eee')
    ax1.set_xlabel('Extraction Rate')
    ax1.set_ylabel('Total Keyframes')
    ax1.set_title('Total Keyframes at Different Extraction Rates')
    
    # Estimated storage per extraction rate
    ax2 = fig.add_subplot(gs[0, 1])
    # Assume avg keyframe ~80KB (JPEG at 1080p)
    avg_keyframe_kb = 80
    storage_gb = [total * avg_keyframe_kb / (1024 * 1024) for total in total_keyframes]
    bars = ax2.bar([f'{r} fps' for r in extraction_rates], storage_gb,
                   color=[GRADIENT_CMAP(i/len(extraction_rates)) for i in range(len(extraction_rates))],
                   alpha=0.85, edgecolor='#333')
    for bar, size in zip(bars, storage_gb):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{size:.1f}GB', ha='center', fontsize=9, color='#eee')
    ax2.set_xlabel('Extraction Rate')
    ax2.set_ylabel('Estimated Storage (GB)')
    ax2.set_title(f'Estimated Storage (@{avg_keyframe_kb}KB/frame JPEG)')
    
    # Duration vs FPS scatter (identify clusters)
    ax3 = fig.add_subplot(gs[1, 0])
    sizes_for_scatter = [m['file_size_mb'] for m in metadata_list]
    scatter = ax3.scatter(durations, fps_vals, c=sizes_for_scatter, cmap=GRADIENT_CMAP, 
                         alpha=0.6, s=30, edgecolors='#555', linewidth=0.5)
    plt.colorbar(scatter, ax=ax3, label='File Size (MB)')
    ax3.set_xlabel('Duration (seconds)')
    ax3.set_ylabel('FPS')
    ax3.set_title('Duration vs FPS (colored by size)')
    
    # Videos segmented by duration buckets
    ax4 = fig.add_subplot(gs[1, 1])
    duration_buckets = {
        '< 10 min': (0, 600),
        '10-15 min': (600, 900),
        '15-20 min': (900, 1200),
        '20-25 min': (1200, 1500),
        '25-30 min': (1500, 1800),
        '> 30 min': (1800, float('inf'))
    }
    bucket_counts = []
    bucket_labels = []
    for label, (lo, hi) in duration_buckets.items():
        count = sum(1 for d in durations if lo <= d < hi)
        bucket_counts.append(count)
        bucket_labels.append(label)
    
    wedges, texts, autotexts = ax4.pie(
        bucket_counts, labels=bucket_labels, autopct=lambda pct: f'{pct:.1f}%\n({int(pct*len(durations)/100)})',
        colors=[GRADIENT_CMAP(i/len(bucket_labels)) for i in range(len(bucket_labels))],
        textprops={'color': '#eee', 'fontsize': 9},
        startangle=90
    )
    ax4.set_title('Videos by Duration Bucket')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_report(metadata_list, brightness_data, output_path):
    """Generate a comprehensive text report with preprocessing recommendations."""
    durations = [m['duration_sec'] for m in metadata_list]
    sizes = [m['file_size_mb'] for m in metadata_list]
    frames = [m['nb_frames'] for m in metadata_list if m['nb_frames'] > 0]
    fps_vals = [m['fps'] for m in metadata_list]
    codecs = Counter(m['codec'] for m in metadata_list)
    resolutions = Counter(m['resolution'] for m in metadata_list)
    groups = sorted(set(m['group'] for m in metadata_list))
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  HCMAI25 VIDEO DATASET - EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # ── Dataset Overview ──
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total videos:         {len(metadata_list)}\n")
        f.write(f"  Number of groups:     {len(groups)} ({', '.join(groups)})\n")
        f.write(f"  Total duration:       {sum(durations)/3600:.2f} hours\n")
        f.write(f"  Total size:           {sum(sizes)/1024:.2f} GB\n")
        f.write(f"  Total frames:         {sum(frames):,}\n")
        f.write(f"  Videos with audio:    {sum(1 for m in metadata_list if m['has_audio'])}/{len(metadata_list)}\n\n")
        
        # ── Duration Statistics ──
        f.write("2. DURATION STATISTICS\n")
        f.write("-" * 40 + "\n")
        dur_min = [d/60 for d in durations]
        f.write(f"  Mean:    {np.mean(dur_min):.2f} min\n")
        f.write(f"  Median:  {np.median(dur_min):.2f} min\n")
        f.write(f"  Std Dev: {np.std(dur_min):.2f} min\n")
        f.write(f"  Min:     {min(dur_min):.2f} min\n")
        f.write(f"  Max:     {max(dur_min):.2f} min\n")
        f.write(f"  P25:     {np.percentile(dur_min, 25):.2f} min\n")
        f.write(f"  P75:     {np.percentile(dur_min, 75):.2f} min\n")
        f.write(f"  P90:     {np.percentile(dur_min, 90):.2f} min\n\n")
        
        # ── Resolution & FPS ──
        f.write("3. RESOLUTION & FPS\n")
        f.write("-" * 40 + "\n")
        f.write("  Resolutions:\n")
        for res, count in resolutions.most_common():
            f.write(f"    {res}: {count} videos ({count/len(metadata_list)*100:.1f}%)\n")
        f.write("\n  FPS values:\n")
        fps_counter = Counter(fps_vals)
        for fps, count in sorted(fps_counter.items()):
            f.write(f"    {fps} fps: {count} videos ({count/len(metadata_list)*100:.1f}%)\n")
        f.write("\n")
        
        # ── Codec & Size ──
        f.write("4. CODEC & FILE SIZE\n")
        f.write("-" * 40 + "\n")
        f.write("  Codecs:\n")
        for codec, count in codecs.most_common():
            codec_sizes = [m['file_size_mb'] for m in metadata_list if m['codec'] == codec]
            f.write(f"    {codec}: {count} videos, avg size={np.mean(codec_sizes):.1f} MB\n")
        f.write(f"\n  File size: mean={np.mean(sizes):.1f} MB, median={np.median(sizes):.1f} MB\n")
        f.write(f"  Bitrate: mean={np.mean([m['bitrate_kbps'] for m in metadata_list]):.0f} kbps\n\n")
        
        # ── Visual Quality (if sampled) ──
        if brightness_data:
            f.write("5. VISUAL QUALITY (Sampled)\n")
            f.write("-" * 40 + "\n")
            br = [d['mean_brightness'] for d in brightness_data]
            ct = [d['mean_contrast'] for d in brightness_data]
            st = [d['mean_saturation'] for d in brightness_data]
            f.write(f"  Brightness: mean={np.mean(br):.1f}, std={np.std(br):.1f}, range=[{min(br):.1f}, {max(br):.1f}]\n")
            f.write(f"  Contrast:   mean={np.mean(ct):.1f}, std={np.std(ct):.1f}, range=[{min(ct):.1f}, {max(ct):.1f}]\n")
            f.write(f"  Saturation: mean={np.mean(st):.1f}, std={np.std(st):.1f}, range=[{min(st):.1f}, {max(st):.1f}]\n")
            
            dark_videos = [d for d in brightness_data if d['mean_brightness'] < 50]
            bright_videos = [d for d in brightness_data if d['mean_brightness'] > 200]
            low_contrast = [d for d in brightness_data if d['mean_contrast'] < 30]
            f.write(f"  Dark videos (brightness < 50):     {len(dark_videos)}/{len(brightness_data)}\n")
            f.write(f"  Bright videos (brightness > 200):  {len(bright_videos)}/{len(brightness_data)}\n")
            f.write(f"  Low contrast (std < 30):           {len(low_contrast)}/{len(brightness_data)}\n\n")
        
        # ── Per-Group Summary ──
        f.write("6. PER-GROUP SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Group':<8} {'#Vids':<7} {'TotalDur(h)':<14} {'AvgDur(m)':<12} {'TotalSize(GB)':<15} {'Codecs'}\n")
        f.write(f"  {'─'*8} {'─'*7} {'─'*14} {'─'*12} {'─'*15} {'─'*20}\n")
        for g in groups:
            gdata = [m for m in metadata_list if m['group'] == g]
            gdur = [m['duration_sec'] for m in gdata]
            gsize = [m['file_size_mb'] for m in gdata]
            gcodecs = Counter(m['codec'] for m in gdata)
            codec_str = ', '.join(f"{c}:{n}" for c, n in gcodecs.most_common())
            f.write(f"  {g:<8} {len(gdata):<7} {sum(gdur)/3600:<14.2f} {np.mean(gdur)/60:<12.2f} {sum(gsize)/1024:<15.2f} {codec_str}\n")
        f.write("\n")
        
        # ── Preprocessing Recommendations ──
        f.write("=" * 80 + "\n")
        f.write("  PREPROCESSING RECOMMENDATIONS FOR 4 TASKS\n")
        f.write("=" * 80 + "\n\n")
        
        # Uniform FPS check
        unique_fps = set(fps_vals)
        mixed_fps = len(unique_fps) > 1
        
        # Common recommendations
        f.write("A. COMMON PREPROCESSING (All Tasks)\n")
        f.write("-" * 40 + "\n")
        
        if mixed_fps:
            f.write(f"  ⚠ MIXED FPS DETECTED: {sorted(unique_fps)}\n")
            f.write(f"    → Normalize all videos to a uniform FPS (recommend {min(unique_fps)} fps)\n")
            f.write(f"    → Or extract keyframes at a fixed interval (e.g., 1-2 fps)\n\n")
        
        if len(resolutions) > 1:
            f.write(f"  ⚠ MIXED RESOLUTIONS: {dict(resolutions)}\n")
            f.write(f"    → Standardize resolution. Most common: {resolutions.most_common(1)[0][0]}\n\n")
        
        if len(codecs) > 1:
            f.write(f"  ⚠ MIXED CODECS: {dict(codecs)}\n")
            f.write(f"    → Be aware that different codecs may affect decoding speed\n")
            f.write(f"    → av1 decoding is slower than h264; consider re-encoding if needed\n\n")
        
        f.write("  General Steps:\n")
        f.write("    1. Extract keyframes at 1-2 fps for indexing\n")
        f.write(f"       Est. keyframes @1fps: {sum(durations):,.0f} frames\n")
        f.write(f"       Est. storage @80KB/frame: {sum(durations)*80/1024/1024:.1f} GB\n")
        f.write("    2. Generate frame-level embeddings (CLIP, SigLIP, etc.)\n")
        f.write("    3. Extract OCR text from keyframes (for text-based queries)\n")
        f.write("    4. Generate captions per keyframe or shot\n")
        f.write("    5. Perform shot boundary detection for scene segmentation\n")
        f.write("    6. Build video-level and frame-level metadata index\n\n")
        
        f.write("B. KIS — Known-Item Search\n")
        f.write("-" * 40 + "\n")
        f.write("  Goal: Find ONE exact scene from natural language description.\n")
        f.write("  Preprocessing focus:\n")
        f.write("    1. Dense keyframe extraction (1-2 fps) to not miss the target\n")
        f.write("    2. High-quality CLIP/SigLIP embeddings for text-to-image matching\n")
        f.write("    3. OCR extraction for visible text matching\n")
        f.write("    4. Object detection (people, vehicles, etc.) for attribute filtering\n")
        f.write("    5. Scene/location classification for context matching\n")
        f.write("    6. Temporal context: store before/after frame relationships\n")
        f.write("    7. Color histogram features for color-based queries\n\n")
        
        f.write("C. AVS — Ad-hoc Video Search\n")
        f.write("-" * 40 + "\n")
        f.write("  Goal: Find ALL scenes matching a general concept.\n")
        f.write("  Preprocessing focus:\n")
        f.write("    1. Shot boundary detection to group frames into shots\n")
        f.write("    2. Shot-level embeddings (average/max pool frame embeddings)\n")
        f.write("    3. Action recognition per shot for activity queries\n")
        f.write("    4. Diversity: de-duplicate near-identical frames within shots\n")
        f.write("    5. Scene classification for location/context filtering\n")
        f.write("    6. Multi-modal fusion: combine visual + text (OCR + caption) features\n\n")
        
        f.write("D. VQA — Video Question Answering\n")
        f.write("-" * 40 + "\n")
        f.write("  Goal: Answer questions grounded in video content.\n")
        f.write("  Preprocessing focus:\n")
        f.write("    1. Dense captions per keyframe for grounding answers\n")
        f.write("    2. Object detection + counting for 'how many' questions\n")
        f.write("    3. Action recognition for 'what does X do' questions\n")
        f.write("    4. OCR for visible text questions\n")
        f.write("    5. Temporal ordering: track event sequences within videos\n")
        f.write("    6. Color/attribute detection for visual property questions\n")
        f.write("    7. Audio transcription (ASR) if questions reference spoken content\n")
        f.write(f"       → {sum(1 for m in metadata_list if m['has_audio'])}/{len(metadata_list)} videos have audio\n\n")
        
        f.write("E. KISC — Known-Item Search with Clarification\n")
        f.write("-" * 40 + "\n")
        f.write("  Goal: Iteratively narrow down to ONE scene through questions.\n")
        f.write("  Preprocessing focus:\n")
        f.write("    1. ALL preprocessing from KIS (same retrieval foundation)\n")
        f.write("    2. Rich attribute annotations per frame for generating questions\n")
        f.write("       (vehicle type, person clothing, scene type, time of day, etc.)\n")
        f.write("    3. Hierarchical scene descriptions for progressive refinement\n")
        f.write("    4. Distinguishing attributes: compute features that differentiate\n")
        f.write("       similar-looking scenes (useful for clarification questions)\n")
        f.write("    5. Cluster similar scenes to identify ambiguous result sets\n\n")
        
        # ── Data Quality Alerts ──
        f.write("=" * 80 + "\n")
        f.write("  DATA QUALITY ALERTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Outlier detection
        dur_mean = np.mean(durations)
        dur_std = np.std(durations)
        outliers = [m for m in metadata_list if abs(m['duration_sec'] - dur_mean) > 2 * dur_std]
        if outliers:
            f.write(f"  ⚠ Duration outliers (>2σ from mean): {len(outliers)} videos\n")
            for o in outliers[:10]:
                f.write(f"    {o['filename']}: {o['duration_min']:.1f} min\n")
            f.write("\n")
        
        # Very short videos
        short_videos = [m for m in metadata_list if m['duration_sec'] < 300]
        if short_videos:
            f.write(f"  ⚠ Very short videos (<5 min): {len(short_videos)}\n")
            for sv in short_videos[:10]:
                f.write(f"    {sv['filename']}: {sv['duration_min']:.1f} min\n")
            f.write("\n")
        
        # Very long videos
        long_videos = [m for m in metadata_list if m['duration_sec'] > 1800]
        if long_videos:
            f.write(f"  ⚠ Very long videos (>30 min): {len(long_videos)}\n")
            for lv in long_videos[:10]:
                f.write(f"    {lv['filename']}: {lv['duration_min']:.1f} min\n")
            f.write("\n")
        
        if brightness_data:
            dark = [d for d in brightness_data if d['mean_brightness'] < 50]
            if dark:
                f.write(f"  ⚠ Very dark videos (brightness < 50): {len(dark)}\n")
                for d in dark:
                    f.write(f"    {d['filename']}: brightness={d['mean_brightness']:.1f}\n")
                f.write("    → Consider brightness normalization or histogram equalization\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  Saved report: {output_path}")


# ── Main Execution ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HCMAI25 Video Dataset - Exploratory Data Analysis")
    print("=" * 60)
    
    # Step 1: Collect metadata
    print("\n[Step 1/4] Collecting video metadata...")
    metadata_list = collect_all_metadata(VIDEO_DIR)
    print(f"  Collected metadata for {len(metadata_list)} videos")
    
    # Save metadata CSV
    csv_path = os.path.join(OUTPUT_DIR, "video_metadata.csv")
    save_metadata_csv(metadata_list, csv_path)
    
    # Step 2: Sample brightness/contrast
    print("\n[Step 2/4] Sampling visual quality metrics...")
    random.seed(42)
    brightness_data = sample_brightness_contrast(
        VIDEO_DIR, metadata_list, BRIGHTNESS_SAMPLE_VIDEOS, SAMPLE_FRAMES_PER_VIDEO)
    
    # Step 3: Generate visualizations
    print("\n[Step 3/4] Generating visualizations...")
    plot_duration_distribution(metadata_list, os.path.join(OUTPUT_DIR, "01_duration_distribution.png"))
    plot_resolution_fps(metadata_list, os.path.join(OUTPUT_DIR, "02_resolution_fps.png"))
    plot_filesize_codec(metadata_list, os.path.join(OUTPUT_DIR, "03_filesize_codec.png"))
    plot_frames_analysis(metadata_list, os.path.join(OUTPUT_DIR, "04_frames_analysis.png"))
    plot_group_statistics(metadata_list, os.path.join(OUTPUT_DIR, "05_group_statistics.png"))
    plot_brightness_contrast(brightness_data, os.path.join(OUTPUT_DIR, "06_brightness_contrast.png"))
    plot_correlation_matrix(metadata_list, os.path.join(OUTPUT_DIR, "07_correlation_matrix.png"))
    plot_temporal_density(metadata_list, os.path.join(OUTPUT_DIR, "08_temporal_density.png"))
    
    # Step 4: Generate report
    print("\n[Step 4/4] Generating EDA report...")
    generate_report(metadata_list, brightness_data, os.path.join(OUTPUT_DIR, "eda_report.txt"))
    
    print("\n" + "=" * 60)
    print("  EDA Complete! All outputs saved to:")
    print(f"    {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
