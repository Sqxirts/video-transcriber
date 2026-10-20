#!/bin/bash
set -euo pipefail

# Service user for transcription. No sudo: nothing in the pipeline needs root.
NEW_USER="transcriber"
PUBKEY="ssh-ed25519 AAAA...your_public_key_here... your-key-comment"

adduser --disabled-password --gecos "" "$NEW_USER"

mkdir -p "/home/$NEW_USER/.ssh"
echo "$PUBKEY" > "/home/$NEW_USER/.ssh/authorized_keys"
chmod 700 "/home/$NEW_USER/.ssh"
chmod 600 "/home/$NEW_USER/.ssh/authorized_keys"
chown -R "$NEW_USER:$NEW_USER" "/home/$NEW_USER/.ssh"

apt-get update
apt-get install -y openssh-server ufw fail2ban

# Drop-in instead of editing sshd_config, so package updates can't overwrite it.
# The file name sorts first, and sshd keeps the first value it reads.
cat > /etc/ssh/sshd_config.d/10-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
EOF

# Refuse to restart on a broken config, or SSH won't come back.
sshd -t
systemctl restart ssh
systemctl enable ssh

ufw allow OpenSSH
ufw --force enable

systemctl enable fail2ban
systemctl start fail2ban

apt-get install -y ffmpeg python3-pip python3-venv

sudo -u "$NEW_USER" bash -c '
    python3 -m venv ~/venv
    source ~/venv/bin/activate
    pip install --upgrade pip
    pip install faster-whisper yt-dlp
'

mkdir -p /home/$NEW_USER/work/input /home/$NEW_USER/work/output
chown -R "$NEW_USER:$NEW_USER" "/home/$NEW_USER/work"

echo ""
echo "Done. SSH in as: ssh -i transcriber_key $NEW_USER@<container-ip>"
echo "Activate the venv with: source ~/venv/bin/activate"
