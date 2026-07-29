# Video Transcriber

## What it is

A Python tool that downloads video/audio (via `yt-dlp`) and transcribes it locally using `faster-whisper`, running in a dedicated, hardened Proxmox LXC container.

## Why I built it

Built this to turn long-form cybersecurity and networking course videos into searchable text notes for studying (CompTIA Network+/Security+, capstone prep, etc.) without relying on a third-party transcription API or uploading course content to the cloud.

## Tech stack

- Python 3.12, `faster-whisper`, `yt-dlp`, `ffmpeg`
- Proxmox LXC container (Ubuntu 24.04), provisioned by a dedicated setup script
- Dedicated non-root user with SSH key-only access

## What it demonstrates

- **Least-privilege provisioning**: `lxc-setup.sh` creates a dedicated service user, disables password auth and root login over SSH, and restricts access to a single authorized key.
- **Container isolation**: runs in its own LXC, separate from other homelab services, with no unnecessary network exposure.
- **Local-first processing**: transcription happens entirely on-box — no third-party API calls, no data leaving the homelab.
- **Reproducible provisioning**: the entire container setup (user creation, SSH hardening, dependency installation) is captured in one idempotent-ish shell script.

## Setup

1. Provision a fresh LXC container and run `lxc-setup.sh` as root (edit `PUBKEY` first to use your own key):
   ```bash
   bash lxc-setup.sh
   ```
2. SSH in as the new user, activate the venv, and run:
   ```bash
   source ~/venv/bin/activate
   python transcribe.py <video-url-or-path>
   ```

## Screenshots / diagrams

None yet.
