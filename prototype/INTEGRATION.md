Integrating the IntegraPose Clipper into Paryamiel/prototype
This guide walks you through adding the IntegraPose Video Clipper GUI to your existing 5-class YOLO+LSTM exam behavior prototype repo. The clipper replaces the manual "record and trim videos into class folders" step (currently Step 1 of your README) with a fast, keyboard-driven labeling tool.

Why the Clipper Fits Your Repo Perfectly
Your repo's current data pipeline:

text

[Manual: Record + trim videos]  →  data/raw_videos/<class>/*.mp4
                                          │
                                          ▼
                              extract_dataset.py
                                          │
                                          ▼
                              data/processed/exam_dataset.npz
                                          │
                                          ▼
                                    train.py
The clipper replaces the manual trimming step with a GUI:

text

[Long recording .mp4]  →  clipper.py  →  data/raw_videos/<class>/*.mp4
                                              │
                                              ▼  (unchanged)
                                    extract_dataset.py
                                              │
                                              ▼
                                    data/processed/exam_dataset.npz
No changes to extract_dataset.py, config.py, train.py, or any other existing file. The clipper outputs directly to data/raw_videos/<class>/, which is exactly what your existing pipeline expects.

File Placement
Add two new files to your repo root:

text

prototype/
├── clipper.py                ← NEW (from this folder)
├── clipper_config.json       ← NEW (from this folder)
├── config.py                 (unchanged)
├── model.py                  (unchanged)
├── dataset.py                (unchanged)
├── tracker.py                (unchanged)
├── extract_dataset.py        (unchanged)
├── train.py                  (unchanged)
├── inference.py              (unchanged)
├── test_prototype.py         (unchanged)
├── requirements.txt          (unchanged — but see "Dependencies" below)
├── data/
│   ├── raw_videos/           ← clipper writes here
│   │   ├── normal/           (auto-created by clipper)
│   │   ├── hand_signal/      (auto-created by clipper)
│   │   ├── passing_of_notes/ (auto-created by clipper)
│   │   ├── side_glancing/    (auto-created by clipper)
│   │   └── use_of_unauthorized_object/ (auto-created by clipper)
│   └── processed/            (unchanged)
├── weights/                  (unchanged)
└── reports/                  (unchanged)
Step-by-Step Integration
1. Download the two files
bash

cd /path/to/your/prototype
# Download both files into your repo root
curl -O https://raw.githubusercontent.com/.../clipper.py            # or copy from this folder
curl -O https://raw.githubusercontent.com/.../clipper_config.json   # or copy from this folder
Or just copy clipper.py and clipper_config.json from this folder into your repo root.

2. Verify dependencies
Your existing requirements.txt likely already includes opencv-python and numpy. The clipper also needs pillow (PIL). If it's not in your requirements, add it:

bash

pip install pillow
Add this line to your requirements.txt:

text

pillow>=10.0.0
3. Update .gitignore (recommended)
The clipper will write video files to data/raw_videos/. If data/raw_videos/ is already in your .gitignore (it should be, since you don't want to commit large video files to git), no change needed.

If not, add:

text

data/raw_videos/
4. Update your README.md (optional but recommended)
Insert a new "Step 1: Label and Trim Raw Videos" section that replaces the current manual trimming instructions. Suggested wording:

STEP 1: Label and Trim Raw Videos (clipper.py)
Use the integrated video clipper GUI to label behavior segments in your longrecordings. The clipper outputs short, labeled .mp4 files directly intodata/raw_videos/<class>/.

```powershellpython clipper.py```

Workflow:
Click Load Video → select a long recording (.mp4 / .avi / .mov).
Use the slider or << / >> buttons to scrub to a behavior segment.
Press one of the labeling keys to pick a class:
n = normal
h = hand_signal
p = passing_of_notes
s = side_glancing
u = use_of_unauthorized_object
Click Set Start Frame at the segment start.
Scrub forward to the segment end, click Set End Frame.
Click Add Clip to Queue.
Repeat steps 2–6 for every behavior segment in this recording.
Click Extract All Queued Clips → batch-exports to data/raw_videos/<class>/.
Keyboard shortcuts:
Space — play / pause
n / h / p / s / u — select behavior class
Delete or Backspace — remove selected clip from queue
How to Use the Clipper — Workflow Summary
Initial run
bash

cd /path/to/your/prototype
python clipper.py
The GUI opens. The first time, data/raw_videos/ and its 5 class subdirectories are auto-created.

For each long recording
Click Load Video → pick a recording (e.g., session01_cam1.mp4, ~15 minutes long).
Scrub to the first behavior segment using the slider or << / >> buttons. Use Space to play/pause for fine scrubbing.
Press one of the keys:
n → behavior = normal
h → behavior = hand_signal
p → behavior = passing_of_notes
s → behavior = side_glancing
u → behavior = use_of_unauthorized_object
At the segment start frame, click Set Start Frame (top-right of clipping controls shows Start: <N>).
Scrub forward to the segment end frame, click Set End Frame (shows End: <N>).
Click Add Clip to Queue → the segment is added to the listbox below.
Repeat steps 2–6 for every behavior segment in this recording.
When done, click Extract All Queued Clips → batch-extracts all queued segments as separate .mp4 files into data/raw_videos/<behavior>/.
Recommended clip lengths
30–60 seconds per segment is the sweet spot.
At 30 fps, that's 900–1800 frames per clip.
With your existing extract_dataset.py (T=30, stride=5), each 30-second clip yields ~540 sliding windows. You'll quickly build a sizable dataset.
Avoid clips shorter than 5 seconds — too few windows per clip.
Output filename convention
The clipper saves files with this naming pattern:

text

<video_stem>_<Behavior>_frames_<start>_<end>.mp4
Examples:

text

data/raw_videos/side_glancing/session01_cam1_side_glancing_frames_2400_3300.mp4
data/raw_videos/normal/session01_cam1_normal_frames_0_900.mp4
data/raw_videos/hand_signal/session02_cam1_hand_signal_frames_1500_1800.mp4
If a filename already exists, the clipper auto-appends _1, _2, etc. to prevent overwrites.

What to do after labeling
Once you've labeled all your recordings:

bash

# Run your existing extraction pipeline (unchanged)
python extract_dataset.py

# Train your LSTM (unchanged)
python train.py --epochs 30 --batch-size 32

# Run real-time inference (unchanged)
python inference.py --source path/to/exam_video.mp4
The clipper's output is 100% compatible with your existing extract_dataset.py — no code changes required downstream.

Attribution Requirements (AGPL-3.0)
The clipper is adapted from IntegraPose (AGPL-3.0). For your thesis:

Keep the attribution header at the top of clipper.py (already included).
Cite the source in your thesis Chapter 2 / References:
Augustine, F., et al. (2025). Integrapose: A Unified Framework for Simultaneous Pose Estimation and Behavior Classification. https://doi.org/10.5281/zenodo.15565090
Repository: https://github.com/farhanaugustine/IntegraPose (commit eb644f1)
License your repo's clipper portion under AGPL-3.0. Since your existing repo doesn't specify a license, the safest path is to add a clipper_LICENSE file (copy from the IntegraPose repo) that applies AGPL-3.0 ONLY to clipper.py. The rest of your repo can stay under whatever license you choose.
You do NOT need to AGPL-license your entire repo — only clipper.py (the adapted file). The other files (your own extract_dataset.py, train.py, etc.) are your original work and can use any license.
For a BS-CS thesis that produces a model + thesis document (no public hosting), AGPL-3.0 is completely fine. You can adapt, modify, and run the code internally without restriction.

Troubleshooting
"Config Error: Missing required key in clipper_config.json"
Make sure clipper_config.json is in the same directory as clipper.py, or pass the path explicitly:

bash

python clipper.py --config /path/to/clipper_config.json
"No module named 'PIL'"
Install pillow:

bash

pip install pillow
Video won't load / playback is choppy
Try a different codec. The clipper uses OpenCV's VideoCapture which supports most formats, but some H.265/HEVC videos may not decode. Re-encode your recordings to H.264 if needed:

bash

ffmpeg -i input.mp4 -c:v libx264 -crf 18 -preset slow output.mp4
Clips don't extract (saved_count = 0)
Check that your video file path doesn't contain non-ASCII characters. OpenCV on Windows can fail silently with Unicode paths. Move the video to a path like C:\videos\session01.mp4 and try again.

Keys don't respond
Click on the video canvas first — keyboard events only fire when the canvas or root window has focus. Don't click on the slider or listbox, or key events won't reach the app.

What's Next
Once the clipper is integrated and you've labeled your first batch of recordings:

Run python extract_dataset.py to extract pose sequences from data/raw_videos/<class>/ into data/processed/exam_dataset.npz.
Run python train.py --epochs 30 --batch-size 32 to train the LSTM.
Run python test_prototype.py to verify the full pipeline.
Run python inference.py --source path/to/test_video.mp4 to test on held-out footage.