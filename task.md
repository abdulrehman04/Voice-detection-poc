# Voice Activity Detection & Speaker Verification Optimization Task

## Overview
This repository contains a Proof of Concept (POC) for a real-time speaker verification and voice activity detection (VAD) system. The system processes an incoming audio stream and determines whether the active speaker matches a target user profile while filtering out silence and background noise.

The current implementation uses **Resemblyzer** and is designed to run efficiently on **CPU** to minimize operational costs. We are looking for an engineer to review this implementation, identify bottlenecks, and propose or implement improvements conform to our constraints.

## Goal
**Improve the quality, robustness, and speed of the voice activity detection and speaker verification pipeline.**
Your objective is to either:
1.  **Optimize the existing solution:** Improve Voice Activity Detection (VAD) accuracy (e.g., handling background noise, end of speech) and speaker verification reliability.
2.  **Propose a better approach:** Demonstrate a more effective method (algorithm/library) that maintains low computational cost (CPU-friendly).

**Constraint:** Do not propose heavy GPU-based solutions unless the performance gain is massive and the cost/latency trade-off is justifiable. We prefer low-latency CPU solutions.

## Current Architecture
- **Core:** Python-based server using `Resemblyzer` for voice embeddings and similarity scoring.
- **Client:** Simple web interface for enrollment and verification only for testing.
- **Processing:** Real-time audio stream analysis.

## Known Issues (The Challenge)
We have identified specific edge cases where the current model struggles:

1.  **Tail Cutoff vs. Background Noise:** 
    - When a user stops speaking, we need a buffer window to avoid cutting the stream too abruptly. 
    - However, during this window, background noise (e.g., TV, other people) can sometimes be incorrectly attributed to the user.
    - *Challenge:* Implement smarter Voice Activity Detection (VAD) or segmentation logic to handle the end-of-speech transition cleanly.

2.  **Short Utterances:**
    - Very short phrases like "yes" or "no" often lack sufficient data for reliable embedding generation.
    - *Challenge:* Improve verification confidence for short audio, segments without increasing false positives.

3.  **Similarity Thresholds:**
    - The current similarity threshold is dynamic, but it requires more rigorous testing across different microphones and environments.
    - *Challenge:* Develop a strategy to make the threshold more robust or adaptive to different recording conditions.

## Task Instructions

1.  **Review the Codebase:** Explore the `poc/` directory to understand the current client-server implementation.
2.  **Reproduce the Issues:** Run the POC and try to replicate the edge cases mentioned above.
3.  **Implement Improvements:**
    - Tweaking code/parameters to fix the "tail" and "short phrase" issues.
    - Replacing components (e.g., VAD, similarity logic) if a better lightweight alternative exists.
4.  **Report / Documentation:**
    - Summary of changes.
    - If you researched other solutions, provide a comparison (Pros/Cons) regarding accuracy vs. CPU usage.

## Setup & Running the POC

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/resemble-ai/Resemblyzer
cd ./Resemblyzer

# Install dependencies
pip install -r requirements_demos.txt
pip install -r poc/server/requirements.txt
```

### 2. Run the Server
The server handles the audio processing and verification logic.
```bash
cd poc/server
python -m contrib.server --port 8765 --debug
```

### 3. Run the Client
Open `poc/client/index.html` in a web browser (or serve it using a simple HTTP server).
1.  Go to `enroll.html` to create a voice profile.
2.  Go to `index.html` to test real-time verification.

---
**Note:** The solution must remain cost-effective. We are targeting a scalable deployment where GPU resources may be too expensive for the base verification layer.
