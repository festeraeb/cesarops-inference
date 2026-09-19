#!/usr/bin/env bash
# Bootstrap pop-os @ 192.168.4.38 as CESARops worker (NFS + Tailscale + NVIDIA).
set -euo pipefail

T440_LAN="${T440_LAN:-192.168.4.29}"
T440_TS="${T440_TS:-100.72.182.77}"
HOSTNAME_NEW="${HOSTNAME_NEW:-cesarops4}"
CESAROPS_PASS="${CESAROPS_PASS:-cesarops}"
MARK="# cesarops worker nfs (bootstrap_pop_os_worker.sh)"

log() { echo "[bootstrap] $*"; }
run_sudo() { echo "$CESAROPS_PASS" | sudo -S "$@"; }

log "hostname -> $HOSTNAME_NEW"
run_sudo hostnamectl set-hostname "$HOSTNAME_NEW"
grep -q "$HOSTNAME_NEW" /etc/hosts || echo "127.0.1.1 $HOSTNAME_NEW" | run_sudo tee -a /etc/hosts

log "packages: nfs, build tools, tailscale deps"
export DEBIAN_FRONTEND=noninteractive
run_sudo apt-get update -qq
run_sudo apt-get install -y -qq nfs-common cifs-utils build-essential dkms linux-headers-"$(uname -r)" curl

log "NFS mount points"
for d in \
  /mnt/t440-models \
  /mnt/t440/repo \
  /mnt/t440/raid0 \
  /mnt/t440/g_armor \
  /mnt/t440/home \
  /mnt/t440/codebase \
  /mnt/t440/data; do
  run_sudo mkdir -p "$d"
done

add_nfs() {
  local src="$1" dst="$2" opts="${3:-vers=3,rw,relatime,rsize=1048576,wsize=1048576,soft,timeo=50,retrans=2,_netdev,nofail}"
  if ! grep -q "$dst" /etc/fstab; then
    echo "$MARK" | run_sudo tee -a /etc/fstab >/dev/null
    echo "${T440_LAN}:${src} ${dst} nfs ${opts} 0 0" | run_sudo tee -a /etc/fstab
  fi
}

add_nfs /mnt/raid0/models /mnt/t440-models "ro,vers=3,soft,timeo=50,retrans=2,nofail,_netdev"
add_nfs /codebase/repos/wreckhunter2000-1 /mnt/t440/repo
add_nfs /mnt/raid0 /mnt/t440/raid0
add_nfs /mnt/g_armor /mnt/t440/g_armor
add_nfs /home/cesarops /mnt/t440/home
add_nfs /codebase /mnt/t440/codebase
add_nfs /data /mnt/t440/data

log "mount -a (nfs)"
run_sudo mount -a || true
df -hT | grep -E 't440|Filesystem' || true

log "Tailscale install"
if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | run_sudo sh
fi
if [ -n "${TAILSCALE_AUTH_KEY:-}" ]; then
  run_sudo tailscale up --authkey="$TAILSCALE_AUTH_KEY" --hostname="$HOSTNAME_NEW" --accept-routes --ssh
else
  log "No TAILSCALE_AUTH_KEY — run: sudo tailscale up --hostname=$HOSTNAME_NEW --accept-routes --ssh"
fi

log "NVIDIA: stub unsupported P106-100 (10de:1c07) so 1070 + P1000 can load"
PCI_STUB_IDS='10de:1c07'
GRUB_FILE=/etc/default/grub
if ! grep -q 'pci-stub.ids' "$GRUB_FILE" 2>/dev/null; then
  run_sudo sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"pci-stub.ids=${PCI_STUB_IDS} /" "$GRUB_FILE"
  run_sudo update-grub
fi
run_sudo mkdir -p /etc/modprobe.d
echo "install pci_stub /sbin/modprobe --ignore-install pci_stub \$CMDLINE_OPTS" | run_sudo tee /etc/modprobe.d/pci-stub.conf >/dev/null
echo "options pci_stub ids=${PCI_STUB_IDS}" | run_sudo tee /etc/modprobe.d/pci-stub-options.conf >/dev/null
run_sudo update-initramfs -u

log "NVIDIA driver already: $(dpkg -l | awk '/^ii.*nvidia-driver/{print $2,$3}' | head -1)"
log "After reboot: nvidia-smi should show GTX 1070 + Quadro P1000 (P106 stubbed)"

log "done — REBOOT required for pci-stub + nvidia"
