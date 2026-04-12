# Flash Talos v1.12.6 ISO to USB

## Schematic ID (NVIDIA open driver extensions)
```
036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2
```

Uses `nvidia-open-gpu-kernel-modules` (required for RTX 5090 / Blackwell GPUs).

---

## Step 1: Download the ISO

```bash
cd /home/david/dev/home-k8s

# Download Talos v1.12.6 with NVIDIA extensions
curl -LO https://factory.talos.dev/image/036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2/v1.12.6/metal-amd64.iso
```

---

## Step 2: Find Your USB Device

```bash
# List all disks (before inserting USB)
lsblk

# Insert your USB drive, then run again
lsblk

# Your USB will be the NEW device (usually /dev/sdb or /dev/sdc)
# MAKE SURE you identify it correctly!
```

---

## Step 3: Flash the ISO to USB

**WARNING: This will ERASE everything on the USB drive!**

```bash
# Replace /dev/sdX with your actual USB device (e.g., /dev/sdb)
sudo dd if=metal-amd64.iso of=/dev/sdX bs=4M status=progress && sync
```

---

## Step 4: Boot Your Server

1. Plug the USB into your server
2. Power on and enter BIOS/boot menu (usually F12, F2, or DEL)
3. Select the USB drive to boot from
4. Talos will boot into maintenance mode
5. **Note the IP address** displayed on screen (DHCP assigned)

---

## Step 5: Save Important Info

Factory image for `controlplane.yaml`:
```
factory.talos.dev/installer/036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2:v1.12.6
```

Schematic file saved at:
```
/home/david/dev/home-k8s/talos-nvidia-schematic.yaml
```

---

## Next Steps

After booting from USB:
1. Generate configs: `talosctl gen config home https://10.0.0.30:6443`
2. Edit `controlplane.yaml` (set install disk to the 256GB SSD, set factory image, add NVIDIA kernel modules)
3. Apply config: `talosctl apply-config --insecure -n <DHCP_IP> --file controlplane.yaml`
4. Bootstrap Kubernetes: `talosctl bootstrap -n <NODE_IP> -e <NODE_IP> --talosconfig ./talosconfig`
5. Get kubeconfig: `talosctl kubeconfig`
6. Bootstrap Flux: `flux bootstrap github --owner dbirks --repository home-k8s --branch main --personal`
