# Video Transcriber

Turns video and audio into text, entirely on my own hardware. Give it a file or a URL; it downloads with `yt-dlp` if needed and transcribes with `faster-whisper`. I built it to turn long course videos into searchable study notes without uploading them to a third-party service.

## What's in the repo

- `transcribe.py`: the pipeline. Resolves a file or URL, runs the speech model, writes a plain-text transcript, and optionally an `.srt` subtitle file. Uses an NVIDIA GPU when one is usable and falls back to CPU otherwise.
- `run-gpu.sh`: GPU launcher. The CUDA libraries pip installs live inside the virtual environment, where the loader doesn't look, so this script adds them to the path first.
- `extract-pptx.py`: pulls slide text and speaker notes out of a PowerPoint file with only the standard library, to pair lecture slides with their transcripts.
- `lxc-setup.sh`: sets up a fresh Ubuntu container to run it.

## How the container is set up

- **A dedicated user with no admin rights.** Transcription doesn't need root, so the `transcriber` user doesn't get sudo.
- **SSH by key only.** Password login and root login are turned off.
- **SSH settings go in a separate drop-in file** (`/etc/ssh/sshd_config.d/10-hardening.conf`), and the script checks the config with `sshd -t` before restarting SSH, so a typo can't lock you out.
- **Firewall on by default.** Only SSH is allowed in (ufw), and repeated failed logins get the address banned (fail2ban).
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

**On a GPU (WSL2 or Linux with an NVIDIA card):** create a `venv` next to the scripts, `pip install faster-whisper yt-dlp nvidia-cublas-cu12 nvidia-cudnn-cu12`, then run `./run-gpu.sh <file-or-url>`. It defaults to the `large-v3` model there; on CPU the default is `small`, because large models are far slower without a GPU.
