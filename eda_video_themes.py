#!/usr/bin/env python3
"""
Video Theme Analysis & Duplicate Detection for HCMAI25 Dataset
================================================================
Analyzes:
  1. Video themes via visual fingerprinting (color histograms, scene types)
  2. Near-duplicate / identical video detection (dHash + histogram correlation)
  3. Cross-group content overlap
  4. Data quality issues (corrupted frames, static videos, etc.)

Outputs:
  - eda_output/theme_analysis/duplicate_report.txt
  - eda_output/theme_analysis/similarity_matrix.png
  - eda_output/theme_analysis/theme_clusters.png
  - eda_output/theme_analysis/duplicate_pairs_*.png (visual comparisons)
  - eda_output/theme_analysis/group_theme_overview.png
  - eda_output/theme_analysis/sample_frames/ (sample grids per group)
  - eda_output/theme_analysis/issues_report.txt
"""

import os
import sys
import csv
import json
import hashlib
import math
import random
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Configuration ──────────────────────────────────────────────────────────
VIDEO_DIR = "/mlcv2025/Datasets/HCMAI25/batch2/video"
OUTPUT_DIR = "/workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026/eda_output/theme_analysis"
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "sample_frames")

FRAMES_PER_VIDEO = 10        # Frames to sample per video for analysis
MAX_WORKERS = 4              # Parallel workers (limited for AV1 decode)
DHASH_SIZE = 16              # dHash size (16x16 = 256-bit hash)
DUPLICATE_HASH_THRESHOLD = 15  # Hamming distance threshold for near-duplicate frames
DUPLICATE_VIDEO_THRESHOLD = 0.85  # Histogram correlation threshold for duplicate videos
SIMILARITY_THRESHOLD = 0.75  # For flagging similar (not duplicate) videos

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
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


# ── Utility Functions ──────────────────────────────────────────────────────

def dhash(image, hash_size=16):
    """Compute difference hash (dHash) of an image."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two boolean hash arrays."""
    return np.sum(hash1 != hash2)


def compute_color_histogram(frame, bins=64):
    """Compute normalized color histogram (HSV) for a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    return hist


def compute_scene_features(frame):
    """Extract scene-level features from a frame."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    features = {}
    # Brightness / contrast
    features['brightness'] = float(np.mean(gray))
    features['contrast'] = float(np.std(gray))
    # Color dominance
    features['saturation'] = float(np.mean(hsv[:, :, 1]))
    features['hue_mean'] = float(np.mean(hsv[:, :, 0]))
    features['hue_std'] = float(np.std(hsv[:, :, 0]))
    # Edge density (complexity proxy)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = float(np.mean(edges) / 255.0)
    # Color ratios (rough scene classification)
    # Blue-ish (sky/water)
    blue_mask = (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] > 40)
    features['blue_ratio'] = float(np.sum(blue_mask)) / (h * w)
    # Green-ish (vegetation)
    green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40)
    features['green_ratio'] = float(np.sum(green_mask)) / (h * w)
    # Dark regions (indoor/night)
    dark_mask = gray < 50
    features['dark_ratio'] = float(np.sum(dark_mask)) / (h * w)
    # Bright regions
    bright_mask = gray > 200
    features['bright_ratio'] = float(np.sum(bright_mask)) / (h * w)
    # Skin-tone (people detection proxy)
    skin_mask = (hsv[:, :, 0] > 0) & (hsv[:, :, 0] < 25) & \
                (hsv[:, :, 1] > 30) & (hsv[:, :, 1] < 180) & \
                (hsv[:, :, 2] > 80)
    features['skin_ratio'] = float(np.sum(skin_mask)) / (h * w)
    # Text-like regions (high contrast small regions)
    # Check top and bottom bars for overlays
    top_bar = gray[:int(h*0.12), :]
    bottom_bar = gray[int(h*0.88):, :]
    features['top_bar_edge'] = float(np.mean(cv2.Canny(top_bar, 50, 150)) / 255.0)
    features['bottom_bar_edge'] = float(np.mean(cv2.Canny(bottom_bar, 50, 150)) / 255.0)

    return features


def classify_scene_type(features):
    """Heuristic scene classification based on visual features."""
    labels = []
    if features['dark_ratio'] > 0.4:
        labels.append('dark/night')
    if features['blue_ratio'] > 0.15:
        labels.append('sky/water')
    if features['green_ratio'] > 0.2:
        labels.append('outdoor/nature')
    if features['skin_ratio'] > 0.1:
        labels.append('people-close')
    if features['edge_density'] > 0.15:
        labels.append('complex/busy')
    elif features['edge_density'] < 0.05:
        labels.append('simple/minimal')
    if features['brightness'] > 160:
        labels.append('bright')
    if features['top_bar_edge'] > 0.1 or features['bottom_bar_edge'] > 0.1:
        labels.append('has-overlay/text')
    if features['saturation'] < 30:
        labels.append('grayscale/muted')
    if not labels:
        labels.append('neutral/indoor')
    return labels


# ── Video Analysis ─────────────────────────────────────────────────────────

def analyze_single_video(video_path, frames_per_video=FRAMES_PER_VIDEO):
    """Analyze a single video: extract frames, compute features."""
    filename = os.path.basename(video_path)
    group = filename.split('_')[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return None

    # Sample frames at evenly-spaced intervals (skip first/last 5%)
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)
    frame_indices = np.linspace(start_frame, end_frame, frames_per_video, dtype=int)

    dhashes = []
    histograms = []
    scene_features_list = []
    scene_labels_all = []
    frames_for_grid = []
    frame_brightnesses = []
    static_count = 0
    prev_gray = None
    issues = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            issues.append(f"Failed to read frame {idx}")
            continue

        # Check for corrupted/blank frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 3:
            issues.append(f"Blank/corrupted frame at {idx} (std={np.std(gray):.1f})")

        # Check for static video (same frame repeated)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            if np.mean(diff) < 1.0:
                static_count += 1
        prev_gray = gray.copy()

        # Compute features
        dhashes.append(dhash(frame, DHASH_SIZE))
        histograms.append(compute_color_histogram(frame))
        sf = compute_scene_features(frame)
        scene_features_list.append(sf)
        scene_labels_all.extend(classify_scene_type(sf))
        frame_brightnesses.append(sf['brightness'])

        # Store small thumbnail for grid
        thumb = cv2.resize(frame, (320, 180))
        frames_for_grid.append(thumb)

    cap.release()

    if not dhashes:
        return None

    # Aggregate features
    avg_histogram = np.mean(histograms, axis=0)
    avg_histogram = avg_histogram / (avg_histogram.sum() + 1e-7)

    # Scene diversity within video
    label_counter = Counter(scene_labels_all)
    dominant_labels = label_counter.most_common(5)

    # Average scene features
    avg_features = {}
    for key in scene_features_list[0]:
        avg_features[key] = np.mean([sf[key] for sf in scene_features_list])

    # Brightness variance (temporal variation)
    brightness_variance = np.std(frame_brightnesses)

    # Static detection
    is_mostly_static = static_count >= len(dhashes) * 0.7

    if is_mostly_static:
        issues.append(f"Mostly static video ({static_count}/{len(dhashes)-1} consecutive similar frames)")

    # Internal diversity: average dhash hamming distance between sampled frames
    internal_distances = []
    for i in range(len(dhashes)):
        for j in range(i+1, len(dhashes)):
            internal_distances.append(hamming_distance(dhashes[i], dhashes[j]))
    avg_internal_diversity = np.mean(internal_distances) if internal_distances else 0

    return {
        'filename': filename,
        'group': group,
        'duration': duration,
        'dhashes': dhashes,
        'avg_histogram': avg_histogram,
        'histograms': histograms,
        'avg_features': avg_features,
        'dominant_labels': dominant_labels,
        'label_counter': label_counter,
        'brightness_variance': brightness_variance,
        'internal_diversity': avg_internal_diversity,
        'is_mostly_static': is_mostly_static,
        'issues': issues,
        'frames_for_grid': frames_for_grid,
        'num_frames_sampled': len(dhashes),
    }


def analyze_all_videos(video_dir, max_workers=MAX_WORKERS):
    """Analyze all videos in parallel."""
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    print(f"Found {len(video_files)} videos to analyze")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_video, str(vf)): vf for vf in video_files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            if (i + 1) % 20 == 0 or i + 1 == len(video_files):
                print(f"  Analyzed {i+1}/{len(video_files)} videos...")

    results.sort(key=lambda x: x['filename'])
    return results


# ── Duplicate Detection ───────────────────────────────────────────────────

def find_duplicate_videos(results):
    """Find duplicate/near-duplicate video pairs using histogram correlation + dHash."""
    n = len(results)
    print(f"\nComputing pairwise similarity for {n} videos...")

    # Compute histogram correlation matrix
    histograms = np.array([r['avg_histogram'] for r in results])
    # Use OpenCV's histogram comparison (correlation method)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            corr = cv2.compareHist(
                histograms[i].astype(np.float32),
                histograms[j].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarity_matrix[i][j] = corr
            similarity_matrix[j][i] = corr

    # Find duplicate pairs
    duplicate_pairs = []     # Very high similarity
    similar_pairs = []       # Notable similarity

    for i in range(n):
        for j in range(i+1, n):
            hist_sim = similarity_matrix[i][j]

            # Also check dHash similarity (best matching frames)
            min_hash_dist = float('inf')
            best_frame_pair = (0, 0)
            for fi, h1 in enumerate(results[i]['dhashes']):
                for fj, h2 in enumerate(results[j]['dhashes']):
                    d = hamming_distance(h1, h2)
                    if d < min_hash_dist:
                        min_hash_dist = d
                        best_frame_pair = (fi, fj)

            # Average dhash distance across all frame pairs
            all_dists = []
            for h1 in results[i]['dhashes']:
                for h2 in results[j]['dhashes']:
                    all_dists.append(hamming_distance(h1, h2))
            avg_hash_dist = np.mean(all_dists)

            pair_info = {
                'video_i': results[i]['filename'],
                'video_j': results[j]['filename'],
                'group_i': results[i]['group'],
                'group_j': results[j]['group'],
                'hist_correlation': hist_sim,
                'min_hash_distance': min_hash_dist,
                'avg_hash_distance': avg_hash_dist,
                'best_frame_pair': best_frame_pair,
                'same_group': results[i]['group'] == results[j]['group'],
            }

            if hist_sim >= DUPLICATE_VIDEO_THRESHOLD and min_hash_dist <= DHASH_SIZE * 2:
                pair_info['type'] = 'DUPLICATE'
                duplicate_pairs.append(pair_info)
            elif hist_sim >= SIMILARITY_THRESHOLD:
                pair_info['type'] = 'SIMILAR'
                similar_pairs.append(pair_info)

    # Sort by similarity
    duplicate_pairs.sort(key=lambda x: x['hist_correlation'], reverse=True)
    similar_pairs.sort(key=lambda x: x['hist_correlation'], reverse=True)

    return similarity_matrix, duplicate_pairs, similar_pairs


# ── Visualization Functions ───────────────────────────────────────────────

def plot_similarity_matrix(similarity_matrix, results, output_path):
    """Plot the video-to-video similarity matrix."""
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.suptitle('Video-to-Video Similarity Matrix (Histogram Correlation)',
                 fontsize=16, fontweight='bold', color='#e94560')

    # Group boundaries
    groups = []
    group_boundaries = [0]
    current_group = results[0]['group']
    for i, r in enumerate(results):
        if r['group'] != current_group:
            group_boundaries.append(i)
            groups.append(current_group)
            current_group = r['group']
    group_boundaries.append(len(results))
    groups.append(current_group)

    im = ax.imshow(similarity_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Histogram Correlation', shrink=0.8)

    # Draw group boundaries
    for b in group_boundaries:
        ax.axhline(b - 0.5, color='#16c79a', linewidth=0.5, alpha=0.7)
        ax.axvline(b - 0.5, color='#16c79a', linewidth=0.5, alpha=0.7)

    # Label groups at midpoints
    for i, g in enumerate(groups):
        mid = (group_boundaries[i] + group_boundaries[i+1]) / 2
        ax.text(-2, mid, g, ha='right', va='center', fontsize=7, color='#16c79a')
        ax.text(mid, -2, g, ha='center', va='bottom', fontsize=7, color='#16c79a', rotation=90)

    ax.set_title(f'{len(results)} Videos × {len(results)} Videos', pad=15)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_theme_analysis(results, output_path):
    """Plot theme/scene type analysis."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Video Theme & Scene Analysis', fontsize=18, fontweight='bold', color='#e94560')

    # 1. Overall scene label distribution
    ax1 = fig.add_subplot(gs[0, 0])
    all_labels = Counter()
    for r in results:
        all_labels.update(r['label_counter'])
    labels_sorted = all_labels.most_common(12)
    label_names = [l[0] for l in labels_sorted]
    label_counts = [l[1] for l in labels_sorted]
    bars = ax1.barh(label_names[::-1], label_counts[::-1],
                    color=[GRADIENT_CMAP(i/len(label_names)) for i in range(len(label_names))],
                    alpha=0.85, edgecolor='#333')
    for bar, count in zip(bars, label_counts[::-1]):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{count}', va='center', fontsize=9, color='#eee')
    ax1.set_xlabel('Occurrences (across all sampled frames)')
    ax1.set_title('Scene Type Distribution (All Videos)')

    # 2. Scene types per group (stacked horizontal bars)
    ax2 = fig.add_subplot(gs[0, 1])
    groups = sorted(set(r['group'] for r in results))
    top_labels = [l[0] for l in all_labels.most_common(6)]
    group_label_data = defaultdict(lambda: defaultdict(int))
    for r in results:
        for label, count in r['label_counter'].items():
            if label in top_labels:
                group_label_data[r['group']][label] += count

    bottom = np.zeros(len(groups))
    for i, label in enumerate(top_labels):
        values = [group_label_data[g][label] for g in groups]
        ax2.barh(groups, values, left=bottom, label=label,
                 color=COLORS[i % len(COLORS)], alpha=0.8, edgecolor='#333')
        bottom += np.array(values)
    ax2.set_xlabel('Count')
    ax2.set_title('Scene Types per Group')
    ax2.legend(fontsize=7, loc='lower right')

    # 3. Internal diversity distribution
    ax3 = fig.add_subplot(gs[1, 0])
    diversities = [r['internal_diversity'] for r in results]
    n, bins, patches = ax3.hist(diversities, bins=30, alpha=0.8, edgecolor='#333')
    for patch, b in zip(patches, bins):
        patch.set_facecolor(GRADIENT_CMAP(b / max(bins) if max(bins) > 0 else 0))
    ax3.axvline(np.mean(diversities), color='#f7b731', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(diversities):.1f}')
    ax3.set_xlabel('Average Internal dHash Distance')
    ax3.set_ylabel('Count')
    ax3.set_title('Video Internal Diversity\n(Higher = more varied content)')
    ax3.legend()

    # 4. Brightness variance (temporal variation)
    ax4 = fig.add_subplot(gs[1, 1])
    bv_per_group = defaultdict(list)
    for r in results:
        bv_per_group[r['group']].append(r['brightness_variance'])
    bp = ax4.boxplot([bv_per_group[g] for g in groups], labels=groups,
                     patch_artist=True, medianprops=dict(color='#f7b731', linewidth=2))
    for patch, color in zip(bp['boxes'], plt.cm.coolwarm(np.linspace(0, 1, len(groups)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xlabel('Group')
    ax4.set_ylabel('Brightness Std Dev')
    ax4.set_title('Temporal Brightness Variation per Group')
    ax4.tick_params(axis='x', rotation=45)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_group_theme_overview(results, output_path):
    """Create a visual overview of average scene features per group."""
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Group-Level Theme Overview', fontsize=18, fontweight='bold', color='#e94560')

    groups = sorted(set(r['group'] for r in results))
    feature_keys = ['brightness', 'contrast', 'saturation', 'edge_density', 'green_ratio', 'skin_ratio']
    feature_titles = ['Avg Brightness', 'Avg Contrast', 'Avg Saturation',
                      'Edge Density (Complexity)', 'Green Ratio (Nature)', 'Skin Ratio (People)']

    for idx, (fkey, ftitle) in enumerate(zip(feature_keys, feature_titles)):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        group_vals = []
        for g in groups:
            vals = [r['avg_features'][fkey] for r in results if r['group'] == g]
            group_vals.append(np.mean(vals))

        bars = ax.bar(groups, group_vals,
                      color=[GRADIENT_CMAP(v / max(group_vals)) if max(group_vals) > 0 else '#e94560'
                             for v in group_vals],
                      alpha=0.85, edgecolor='#333')
        ax.set_title(ftitle, fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_ylabel(fkey)
        # Add mean line
        ax.axhline(np.mean(group_vals), color='#f7b731', linestyle='--', linewidth=1, alpha=0.7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def save_duplicate_comparison(results, pair_info, output_dir, pair_idx):
    """Save visual comparison of a duplicate/similar pair."""
    v_i = next(r for r in results if r['filename'] == pair_info['video_i'])
    v_j = next(r for r in results if r['filename'] == pair_info['video_j'])

    n_frames = min(5, len(v_i['frames_for_grid']), len(v_j['frames_for_grid']))
    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 4, 5))
    fig.suptitle(
        f"{pair_info['type']}: {pair_info['video_i']} vs {pair_info['video_j']}\n"
        f"Hist Corr: {pair_info['hist_correlation']:.3f} | "
        f"Min Hash Dist: {pair_info['min_hash_distance']} | "
        f"Avg Hash Dist: {pair_info['avg_hash_distance']:.1f}",
        fontsize=11, fontweight='bold', color='#e94560'
    )

    indices_i = np.linspace(0, len(v_i['frames_for_grid']) - 1, n_frames, dtype=int)
    indices_j = np.linspace(0, len(v_j['frames_for_grid']) - 1, n_frames, dtype=int)

    for col in range(n_frames):
        # Top row: video i
        frame_i = cv2.cvtColor(v_i['frames_for_grid'][indices_i[col]], cv2.COLOR_BGR2RGB)
        if n_frames > 1:
            axes[0, col].imshow(frame_i)
            axes[0, col].set_title(f'{v_i["filename"]}\nFrame {col+1}', fontsize=8, color='#eee')
            axes[0, col].axis('off')
        else:
            axes[0].imshow(frame_i)
            axes[0].set_title(f'{v_i["filename"]}', fontsize=8, color='#eee')
            axes[0].axis('off')

        # Bottom row: video j
        frame_j = cv2.cvtColor(v_j['frames_for_grid'][indices_j[col]], cv2.COLOR_BGR2RGB)
        if n_frames > 1:
            axes[1, col].imshow(frame_j)
            axes[1, col].set_title(f'{v_j["filename"]}\nFrame {col+1}', fontsize=8, color='#eee')
            axes[1, col].axis('off')
        else:
            axes[1].imshow(frame_j)
            axes[1].set_title(f'{v_j["filename"]}', fontsize=8, color='#eee')
            axes[1].axis('off')

    output_path = os.path.join(output_dir, f'duplicate_pair_{pair_idx:03d}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def save_group_sample_grids(results, output_dir):
    """Save a grid of sample frames for each group."""
    groups = sorted(set(r['group'] for r in results))

    for group in groups:
        group_results = [r for r in results if r['group'] == group]
        # Pick up to 6 videos per group, 3 frames each
        selected = group_results[:6]
        n_vids = len(selected)
        n_frames_show = 3

        fig, axes = plt.subplots(n_vids, n_frames_show, figsize=(n_frames_show * 4, n_vids * 2.5))
        fig.suptitle(f'Group {group} — Sample Frames ({len(group_results)} videos)',
                     fontsize=14, fontweight='bold', color='#e94560')

        for row, r in enumerate(selected):
            indices = np.linspace(0, len(r['frames_for_grid']) - 1, n_frames_show, dtype=int)
            labels = ', '.join([l[0] for l in r['dominant_labels'][:3]])
            for col in range(n_frames_show):
                frame = cv2.cvtColor(r['frames_for_grid'][indices[col]], cv2.COLOR_BGR2RGB)
                if n_vids > 1 and n_frames_show > 1:
                    axes[row, col].imshow(frame)
                    if col == 0:
                        axes[row, col].set_ylabel(r['filename'].replace('.mp4', ''),
                                                   fontsize=7, color='#16c79a')
                    axes[row, col].axis('off')
                    if row == 0:
                        axes[row, col].set_title(f'Frame {col+1}', fontsize=9, color='#eee')
                elif n_vids == 1:
                    axes[col].imshow(frame)
                    axes[col].axis('off')
                elif n_frames_show == 1:
                    axes[row].imshow(frame)
                    axes[row].axis('off')

        plt.savefig(os.path.join(output_dir, f'group_{group}_samples.png'),
                    dpi=100, bbox_inches='tight')
        plt.close()

    print(f"  Saved sample grids for {len(groups)} groups")


def plot_cross_group_similarity(similarity_matrix, results, output_path):
    """Compute and plot average similarity between groups."""
    groups = sorted(set(r['group'] for r in results))
    n_groups = len(groups)
    group_indices = defaultdict(list)
    for i, r in enumerate(results):
        group_indices[r['group']].append(i)

    group_sim = np.zeros((n_groups, n_groups))
    for gi, g1 in enumerate(groups):
        for gj, g2 in enumerate(groups):
            sims = []
            for i in group_indices[g1]:
                for j in group_indices[g2]:
                    if i != j:
                        sims.append(similarity_matrix[i][j])
            group_sim[gi][gj] = np.mean(sims) if sims else 0

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle('Cross-Group Average Similarity', fontsize=16, fontweight='bold', color='#e94560')

    im = ax.imshow(group_sim, cmap='hot', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Avg Histogram Correlation', shrink=0.8)

    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(groups, rotation=45, fontsize=9)
    ax.set_yticklabels(groups, fontsize=9)

    for i in range(n_groups):
        for j in range(n_groups):
            color = 'white' if group_sim[i, j] > 0.5 else '#eee'
            ax.text(j, i, f'{group_sim[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ── Report Generation ─────────────────────────────────────────────────────

def generate_theme_report(results, duplicate_pairs, similar_pairs, output_path):
    """Generate the comprehensive theme & duplicate detection report."""
    groups = sorted(set(r['group'] for r in results))

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  HCMAI25 VIDEO THEME ANALYSIS & DUPLICATE DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # ── Summary ──
        f.write("1. ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Videos analyzed:          {len(results)}\n")
        f.write(f"  Frames sampled/video:     {FRAMES_PER_VIDEO}\n")
        f.write(f"  Duplicate pairs found:    {len(duplicate_pairs)}\n")
        f.write(f"  Similar pairs found:      {len(similar_pairs)}\n")
        f.write(f"  Videos with issues:       {sum(1 for r in results if r['issues'])}\n")
        f.write(f"  Mostly static videos:     {sum(1 for r in results if r['is_mostly_static'])}\n\n")

        # ── Scene Type Overview ──
        f.write("2. SCENE TYPE OVERVIEW (All Videos)\n")
        f.write("-" * 40 + "\n")
        all_labels = Counter()
        for r in results:
            all_labels.update(r['label_counter'])
        for label, count in all_labels.most_common():
            pct = count / sum(all_labels.values()) * 100
            f.write(f"  {label:<25} {count:>6} occurrences ({pct:.1f}%)\n")
        f.write("\n")

        # ── Per-Group Theme Summary ──
        f.write("3. PER-GROUP THEME SUMMARY\n")
        f.write("-" * 40 + "\n")
        for g in groups:
            g_results = [r for r in results if r['group'] == g]
            g_labels = Counter()
            for r in g_results:
                g_labels.update(r['label_counter'])
            top_themes = g_labels.most_common(5)
            theme_str = ', '.join(f"{l[0]}({l[1]})" for l in top_themes)

            avg_bright = np.mean([r['avg_features']['brightness'] for r in g_results])
            avg_edge = np.mean([r['avg_features']['edge_density'] for r in g_results])
            avg_diversity = np.mean([r['internal_diversity'] for r in g_results])

            f.write(f"\n  [{g}] {len(g_results)} videos\n")
            f.write(f"    Themes: {theme_str}\n")
            f.write(f"    Avg brightness={avg_bright:.1f}, edge_density={avg_edge:.3f}, ")
            f.write(f"internal_diversity={avg_diversity:.1f}\n")

        f.write("\n")

        # ── Duplicate Pairs ──
        f.write("=" * 80 + "\n")
        f.write("4. DUPLICATE VIDEO PAIRS\n")
        f.write(f"   (Histogram correlation >= {DUPLICATE_VIDEO_THRESHOLD} AND ")
        f.write(f"min dHash distance <= {DHASH_SIZE * 2})\n")
        f.write("=" * 80 + "\n\n")

        if not duplicate_pairs:
            f.write("  No duplicate pairs detected.\n\n")
        else:
            for i, pair in enumerate(duplicate_pairs):
                f.write(f"  [{i+1}] {pair['video_i']} <-> {pair['video_j']}\n")
                f.write(f"      Groups: {pair['group_i']} / {pair['group_j']} ")
                f.write(f"({'SAME' if pair['same_group'] else 'DIFFERENT'} group)\n")
                f.write(f"      Hist Correlation: {pair['hist_correlation']:.4f}\n")
                f.write(f"      Min dHash Distance: {pair['min_hash_distance']}\n")
                f.write(f"      Avg dHash Distance: {pair['avg_hash_distance']:.1f}\n\n")

        # ── Similar Pairs (top 30) ──
        f.write("=" * 80 + "\n")
        f.write("5. SIMILAR VIDEO PAIRS (Top 30)\n")
        f.write(f"   (Histogram correlation >= {SIMILARITY_THRESHOLD})\n")
        f.write("=" * 80 + "\n\n")

        for i, pair in enumerate(similar_pairs[:30]):
            f.write(f"  [{i+1}] {pair['video_i']} <-> {pair['video_j']}\n")
            f.write(f"      Groups: {pair['group_i']} / {pair['group_j']} ")
            f.write(f"({'SAME' if pair['same_group'] else 'CROSS'} group)\n")
            f.write(f"      Hist Correlation: {pair['hist_correlation']:.4f}\n")
            f.write(f"      Min dHash Distance: {pair['min_hash_distance']}\n\n")

        if len(similar_pairs) > 30:
            f.write(f"  ... and {len(similar_pairs) - 30} more similar pairs.\n\n")

        # Cross-group overlap stats
        cross_group_similar = [p for p in similar_pairs if not p['same_group']]
        same_group_similar = [p for p in similar_pairs if p['same_group']]
        f.write(f"\n  Similar pairs within same group: {len(same_group_similar)}\n")
        f.write(f"  Similar pairs across groups:     {len(cross_group_similar)}\n\n")

        if cross_group_similar:
            f.write("  Cross-group overlap breakdown:\n")
            cross_counter = Counter()
            for p in cross_group_similar:
                key = tuple(sorted([p['group_i'], p['group_j']]))
                cross_counter[key] += 1
            for (g1, g2), count in cross_counter.most_common(20):
                f.write(f"    {g1} <-> {g2}: {count} similar pairs\n")
            f.write("\n")

        # ── Data Quality Issues ──
        f.write("=" * 80 + "\n")
        f.write("6. DATA QUALITY ISSUES\n")
        f.write("=" * 80 + "\n\n")

        # Static videos
        static_vids = [r for r in results if r['is_mostly_static']]
        if static_vids:
            f.write(f"  MOSTLY STATIC VIDEOS ({len(static_vids)}):\n")
            for r in static_vids:
                f.write(f"    {r['filename']} (diversity={r['internal_diversity']:.1f})\n")
            f.write("\n")

        # Low internal diversity
        div_mean = np.mean([r['internal_diversity'] for r in results])
        div_std = np.std([r['internal_diversity'] for r in results])
        low_div = [r for r in results if r['internal_diversity'] < div_mean - 2 * div_std]
        if low_div:
            f.write(f"  LOW DIVERSITY VIDEOS (< 2 std below mean, threshold={div_mean - 2*div_std:.1f}):\n")
            for r in sorted(low_div, key=lambda x: x['internal_diversity']):
                f.write(f"    {r['filename']}: diversity={r['internal_diversity']:.1f}\n")
            f.write("\n")

        # Frame read issues
        issue_vids = [r for r in results if r['issues']]
        if issue_vids:
            f.write(f"  VIDEOS WITH FRAME ISSUES ({len(issue_vids)}):\n")
            for r in issue_vids:
                for issue in r['issues']:
                    f.write(f"    {r['filename']}: {issue}\n")
            f.write("\n")

        if not static_vids and not low_div and not issue_vids:
            f.write("  No data quality issues detected.\n\n")

        f.write("=" * 80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HCMAI25 Video Theme Analysis & Duplicate Detection")
    print("=" * 60)

    # Step 1: Analyze all videos
    print("\n[Step 1/5] Analyzing all videos (this may take a while for AV1)...")
    results = analyze_all_videos(VIDEO_DIR)
    print(f"  Successfully analyzed {len(results)} videos")

    # Step 2: Detect duplicates
    print("\n[Step 2/5] Computing pairwise similarity & detecting duplicates...")
    similarity_matrix, duplicate_pairs, similar_pairs = find_duplicate_videos(results)
    print(f"  Found {len(duplicate_pairs)} duplicate pairs")
    print(f"  Found {len(similar_pairs)} similar pairs")

    # Step 3: Generate visualizations
    print("\n[Step 3/5] Generating visualizations...")
    plot_similarity_matrix(similarity_matrix, results,
                          os.path.join(OUTPUT_DIR, "similarity_matrix.png"))
    plot_theme_analysis(results, os.path.join(OUTPUT_DIR, "theme_clusters.png"))
    plot_group_theme_overview(results, os.path.join(OUTPUT_DIR, "group_theme_overview.png"))
    plot_cross_group_similarity(similarity_matrix, results,
                               os.path.join(OUTPUT_DIR, "cross_group_similarity.png"))

    # Step 4: Save duplicate comparisons and group samples
    print("\n[Step 4/5] Saving comparison images...")
    max_comparisons = min(20, len(duplicate_pairs) + len(similar_pairs))
    all_flagged = duplicate_pairs + similar_pairs[:max(0, max_comparisons - len(duplicate_pairs))]
    for i, pair in enumerate(all_flagged[:20]):
        save_duplicate_comparison(results, pair, OUTPUT_DIR, i)
    print(f"  Saved {min(20, len(all_flagged))} comparison images")

    print("  Saving group sample grids...")
    save_group_sample_grids(results, SAMPLE_DIR)

    # Step 5: Generate report
    print("\n[Step 5/5] Generating report...")
    generate_theme_report(results, duplicate_pairs, similar_pairs,
                         os.path.join(OUTPUT_DIR, "theme_report.txt"))

    print("\n" + "=" * 60)
    print("  Analysis Complete! Outputs saved to:")
    print(f"    {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
