#!/usr/bin/env bash
# Stub P106-100 (10de:1c07) so nvidia-595 can bind GTX 1070 + Quadro P1000.
set -euo pipefail
PASS="${CESAROPS_PASS:-cesarops}"
sudo() { echo "$PASS" | command sudo -S "$@"; }

sudo mkdir -p /etc/modprobe.d
echo 'options pci_stub ids=10de:1c07' | sudo tee /etc/modprobe.d/pci-stub.conf >/dev/null

GRUB=/etc/default/grub
if ! grep -q 'pci-stub.ids=10de:1c07' "$GRUB"; then
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pci-stub.ids=10de:1c07"/' "$GRUB"
fi
sudo update-grub
sudo update-initramfs -u
echo "OK — reboot required: sudo reboot"
