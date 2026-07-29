#!/bin/bash
set -euo pipefail

NEW_USER="claude"
PUBKEY="ssh-ed25519 AAAA...your_public_key_here... your-key-comment"

adduser --disabled-password --gecos "" "$NEW_USER"
usermod -aG sudo "$NEW_USER"

mkdir -p "/home/$NEW_USER/.ssh"
echo "$PUBKEY" > "/home/$NEW_USER/.ssh/authorized_keys"
chmod 700 "/home/$NEW_USER/.ssh"
chmod 600 "/home/$NEW_USER/.ssh/authorized_keys"
chown -R "$NEW_USER:$NEW_USER" "/home/$NEW_USER/.ssh"

echo "$NEW_USER ALL=(ALL) ALL" > "/etc/sudoers.d/$NEW_USER"
chmod 440 "/etc/sudoers.d/$NEW_USER"

apt-get update
apt-get install -y openssh-server
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl restart ssh
systemctl enable ssh

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
