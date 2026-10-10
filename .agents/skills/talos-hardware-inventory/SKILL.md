---
name: talos-hardware-inventory
description: Inspect the physical hardware of the Talos node from the workstation — PCI topology and negotiated link widths, DMI/board identity, which kernel modules exist, block devices and real free disk per storage path — without lspci, lsblk, bash or a shell. Use before buying a GPU/expansion card (does the slot have lanes? does the kernel support Thunderbolt?), when a device "isn't showing up", or when you need to prove how many bytes are free on the NVMe vs the HDD PVCs.
---

# Hardware inventory on an immutable Talos node

Talos has no shell, no `lspci`/`lsblk`, `/` and `/usr` read-only. Everything below runs from
the workstation with `kubectl` + `talosctl`. Set the target once (node IP is **DHCP** — always
re-read it, never hardcode):
```bash
cd ~/dev/home-k8s
NODE_IP=$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
T="talosctl --talosconfig _newconfig/talosconfig -e $NODE_IP -n $NODE_IP"
```

## Board / firmware identity
```bash
for f in sys_vendor product_name bios_version bios_date board_name chassis_type; do
  printf "%-14s %s\n" "$f" "$($T read /sys/class/dmi/id/$f 2>/dev/null | head -1)"; done
```
(`hostname talos-210-73x` is Talos's auto-name from DMI — but read DMI, don't guess from it.)

## PCI topology and REAL link widths (the lspci replacement)
`talosctl list` + `read` over sysfs. Filter out bridges to see actual endpoints; the
`current_link_*` files are what the slot *negotiated*, which is the number that matters:
```bash
for d in $($T list /sys/bus/pci/devices | awk 'NR>1{print $2}'); do
  cls=$($T read /sys/bus/pci/devices/$d/class; ) ; ven=$($T read /sys/bus/pci/devices/$d/vendor)
  dev=$($T read /sys/bus/pci/devices/$d/device); sp=$($T read /sys/bus/pci/devices/$d/current_link_speed)
  wd=$($T read /sys/bus/pci/devices/$d/current_link_width)
  case "$cls" in *0604*|*0601*) continue;; esac      # skip PCI bridges / ISA
  echo "$d class=$cls ven=$ven dev=$dev link=$sp x$wd"
done
```
Reading the output: `class=0x030000 ven=0x10de` = VGA 3D controller; `0x010802` = NVMe;
`0x0c0330` = xHCI; `0x020000`/`0x028000` = NIC/WiFi. A GPU reporting `16.0 GT/s x16` is in
the CPU-attached slot; `8.0 GT/s x1` means it is riding a chipset lane (≈2 GB/s, and it
shares the DMI with the NIC/SATA/USB — see the board spec). Vendor IDs: `0x10de` NVIDIA,
`0x1022` AMD, `0x10ec` Realtek.

## Does the kernel even support the bus you want? (Thunderbolt / USB4 / hotplug)
Two places: the sysfs bus directories, and the module manifests (`modules.builtin` lists
compiled-in modules, `modules.dep` lists loadable ones). Absent from BOTH = not supported,
no extension will save you:
```bash
K=$(kubectl get node -o jsonpath='{.items[0].status.nodeInfo.kernelVersion}')   # NOT uname: that's the workstation
$T list /sys/bus/thunderbolt/devices 2>&1 | tail -1     # "no such file" = no TB bus
$T read /lib/modules/$K/modules.builtin | grep -E "thunderbolt|usb4|pciehp"     # compiled in?
$T read /lib/modules/$K/modules.dep    | grep -E "thunderbolt|usb4|pciehp"     # loadable?
```
On this node (Talos v1.13.6 / kernel 6.18.38): **zero** hits — no Thunderbolt, no USB4, no
PCIe hotplug. A PCIe card must be installed **cold** (power off → seat → boot). Kernel args:
`$T read /proc/cmdline`. Loaded module version: `$T read /sys/module/nvidia/version`.
Extensions present: read the `extensions.talos.dev/*` node labels (`kubectl get node
--show-labels`) — that is the live schematic content; the annotation
`extensions.talos.dev/schematic` is the live schematic ID (verify before trusting any ID
quoted in docs).

## Block devices, mounts, and REAL free space per PVC path
`talosctl mounts` is the one-shot answer — it reports BOTH storage-class filesystems with
free space (kubelet's fs report only covers `/var`):
```bash
$T mounts | grep -E "/var$|hdd-storage"          # size / used / AVAILABLE / mount
$T list /dev/disk/by-id | tail -n +2             # real hardware behind it
```
On this node today: `/dev/nvme0n1p4` → `/var` (Crucial CT1000P510SSD8, ~200 GiB free of 998)
and `/dev/sda1` → `/var/mnt/hdd-storage` (WD40EZZX 4 TB, ~1.9 TiB free). Cross-check a claim's
path in `infra/local-path-provisioner.yaml` (`local-path` → `/var/lib/local-path-provisioner`,
`local-path-hdd` → `/var/mnt/hdd-storage`) and remember local-path does **not** enforce the
`storage:` request — it is a plain hostPath, so `mounts`/`df` is the only truth. Compare free
space against kubelet's DiskPressure eviction (≈10 % free) before staging a big model: that is
the DiskPressure incident the flashnext-radixark and power-cap notes keep referring to.
(If you only have kubectl, the NVMe figure is also in
`kubectl get --raw "/api/v1/nodes/$NODE/proxy/stats/summary"` under `.node.fs`, and the HDD is
visible from any pod mounting a `local-path-hdd` claim: `kubectl exec deploy/jellyfin -- df -h`.)

## GPU specifics
```bash
kubectl exec ds/gpu-power-limit -- nvidia-smi --query-gpu=index,uuid,name,power.limit,persistence_mode --format=csv
kubectl get node -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}'   # what HAMi sees
kubectl get node -o jsonpath='{.status.capacity}' | tr ',' '\n' | grep -E "gpu|memory|cpu"
kubectl top node    # CPU/mem only; includes page cache, so read dips not absolute
```
Distinguish "card is off the bus" from "driver unhappy" with `$T dmesg | grep -iE "nvidia|fullchip|xid"`
(see the gpu-wedge-recovery skill for the recovery path).
