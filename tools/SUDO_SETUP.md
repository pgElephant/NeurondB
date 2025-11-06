# GPU Monitor - Sudo Setup for macOS

To enable real-time GPU monitoring without password prompts, you need to configure passwordless sudo for `powermetrics`.

## Quick Setup

### Option 1: Enable Passwordless Sudo for powermetrics (Recommended)

1. Run this command to edit sudoers file:
```bash
sudo visudo
```

2. Add this line at the end of the file:
```
%admin ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

3. Save and exit (Ctrl+X, then Y, then Enter if using nano, or `:wq` if using vim)

4. Test it:
```bash
sudo -n powermetrics --samplers gpu_power -i 500 -n 1
```

If it runs without asking for password, you're good to go!

### Option 2: Run with sudo each time

Simply run the monitor with sudo and enter your password once:
```bash
sudo ./gpu.py --monitor
```

## Usage

After setup, run:

### With Passwordless Sudo (Option 1)
```bash
./gpu.py --monitor -i 1
```
Shows real GPU utilization percentage!

### With Sudo Prompt (Option 2)
```bash
sudo ./gpu.py --monitor -i 1
```
Enter password once at start.

## What You'll See

With sudo enabled, you'll see:
- **GPU Utilization**: Real-time GPU active percentage (0-100%) ✅
- **GPU Active/Idle**: Breakdown of GPU activity ✅
- **Active Processes**: When GPU >5% active, shows top GPU-capable processes
- Color-coded utilization indicators (green/yellow/red)

**Note**: macOS doesn't provide per-process GPU utilization like NVIDIA. The tool shows:
- Overall GPU utilization (accurate)
- Top active processes when GPU is busy (CPU% as proxy)

Without sudo, you'll see:
- Basic GPU info (chip model)
- Message to run with sudo for GPU metrics

