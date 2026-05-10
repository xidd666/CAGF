"""
Save waveform image for a single audio clip.
Optionally, extract only the segment where a specific word (e.g. "Okay") is spoken.

Usage:
  # Full waveform
  python save_waveform.py --video_id 03bSnISJMiM --clip_id 12 --dataset mosi

  # Waveform cropped to the word "Okay"
  python save_waveform.py --video_id 03bSnISJMiM --clip_id 12 --dataset mosi --word Okay

Requirements for --word:
  pip install openai-whisper
"""

import argparse
import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

audio_map = {
    'mosi':  'data/MOSI/wav',
    'mosei': 'data/MOSEI/wav',
}


def find_word_timestamps(audio_path: str, target_word: str):
    """
    Use Whisper with word_timestamps=True to find every occurrence of
    `target_word` (case-insensitive) in the audio.
    Returns a list of (start_sec, end_sec) tuples.
    """
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")

    matches = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word_text = w["word"].strip().strip(".,!?;:'\"").lower()
            if word_text == target_word.lower():
                matches.append((w["start"], w["end"]))
    return matches


def _plot_waveform(waveform, times, ax):
    cmap = cm.get_cmap('viridis', 4)
    fill_color = cmap(0)

    ax.plot(times, waveform, color=fill_color, linewidth=0.6)
    ax.fill_between(times, waveform, 0, where=None, color=fill_color, alpha=1.0)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    border_lw = 1.2
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(border_lw)
        spine.set_edgecolor('#333333')

    ax.set_xlim(times[0], times[-1])


def _duration_to_figwidth(duration_sec, full_duration_sec, full_width=8.0, min_width=1.5):
    """Scale figure width proportionally to segment duration."""
    return max(min_width, full_width * duration_sec / full_duration_sec)


def save_waveform(video_id, clip_id, audio_dir, target_word=None):
    audio_path = f"{audio_dir}/{video_id}/{clip_id}.wav"
    sound, sr = torchaudio.load(audio_path)
    sr = int(sr) if sr is not None else 16000
    soundData = torch.mean(sound, dim=0, keepdim=False).cpu().numpy()
    times_full = np.arange(soundData.shape[0]) / float(sr)

    full_duration = len(soundData) / float(sr)

    if target_word is None:
        # ── Full waveform ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 3))
        _plot_waveform(soundData, times_full, ax)
        plt.tight_layout()
        img_path = f'{video_id}_{clip_id}_waveform.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved waveform → {img_path}")
    else:
        # ── Word-level crop ────────────────────────────────────────────────
        print(f"Running Whisper to locate \"{target_word}\" in {audio_path} …")
        matches = find_word_timestamps(audio_path, target_word)

        if not matches:
            print(f'Word "{target_word}" not found in {audio_path}.')
            return

        print(f'Found {len(matches)} occurrence(s) of "{target_word}":')
        for i, (t0, t1) in enumerate(matches):
            print(f'  [{i}]  {t0:.3f}s – {t1:.3f}s')

        for i, (t0, t1) in enumerate(matches):
            s0 = int(t0 * sr)
            s1 = int(t1 * sr)
            # add a small 50 ms padding on each side
            pad = int(0.05 * sr)
            s0 = max(0, s0 - pad)
            s1 = min(len(soundData), s1 + pad)

            segment = soundData[s0:s1]
            times_seg = np.arange(segment.shape[0]) / float(sr) + s0 / float(sr)

            seg_duration = len(segment) / float(sr)
            fig_w = _duration_to_figwidth(seg_duration, full_duration)
            fig, ax = plt.subplots(figsize=(fig_w, 3))
            _plot_waveform(segment, times_seg, ax)
            plt.tight_layout()

            suffix = f'_{i}' if len(matches) > 1 else ''
            img_path = f'{video_id}_{clip_id}_{target_word}{suffix}_waveform.png'
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved waveform → {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save waveform image for a single audio clip")
    parser.add_argument('--video_id', type=str, required=True)
    parser.add_argument('--clip_id',  type=int, required=True)
    parser.add_argument('--dataset',  type=str, default='mosi', choices=list(audio_map.keys()))
    parser.add_argument('--word',     type=str, default=None,
                        help='If set, crop waveform to the segment(s) where this word is spoken '
                             '(requires openai-whisper)')
    args = parser.parse_args()

    save_waveform(
        video_id=args.video_id,
        clip_id=args.clip_id,
        audio_dir=audio_map[args.dataset],
        target_word=args.word,
    )
