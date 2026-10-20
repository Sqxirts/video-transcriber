# Video Transcriber

Turns video and audio into text, entirely on my own hardware. Give it a file or a URL; it downloads with `yt-dlp` if needed and transcribes with `faster-whisper`. I built it to turn long course videos into searchable study notes without uploading them to a third-party service.

## What's in the repo

- `transcribe.py`: the pipeline. Resolves a file or URL, runs the speech model, writes a plain-text transcript, and optionally an `.srt` subtitle file.
- `lxc-setup.sh`: sets up a fresh Ubuntu container to run it.

## How the container is set up

- **A dedicated user with no admin rights.** Transcription doesn't need root, so the `transcriber` user doesn't get sudo.
- **SSH by key only.** Password login and root login are turned off.
- **SSH settings go in a separate drop-in file** (`/etc/ssh/sshd_config.d/10-hardening.conf`), and the script checks the config with `sshd -t` before restarting SSH, so a typo can't lock you out.
- **Nothing leaves the machine.** The model runs locally; the only network traffic is the optional download.

## Run it

1. On a fresh Ubuntu 24.04 container, edit `PUBKEY` in `lxc-setup.sh` to your public key, then run it as root:
   ```bash
   bash lxc-setup.sh
   ```
2. SSH in as `transcriber`, then:
   ```bash
   source ~/venv/bin/activate
   python transcribe.py <file-or-url> [--model small] [--srt]
   ```
