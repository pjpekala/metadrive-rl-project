#!/bin/bash
# =============================================================================
# MetaDrive Training Wrapper Script for Headless Servers (e.g., RunPod)
# =============================================================================
# This script configures the environment for proper GPU-accelerated rendering
# on headless servers that don't have a physical display.
#
# Usage:
#   ./run_training.sh train_metadrive.py --modality bev --device cuda
#   ./run_training.sh train_metadrive.py --modality rgb --device cuda
#   ./run_training.sh train_metadrive.py --modality sparse_lidar --device cuda
#
# What this script does:
# 1. Enables NVIDIA graphics capabilities (required for EGL/OpenGL rendering)
# 2. Runs the training script with EGL-based headless rendering (no Xvfb needed)
#
# NOTE: The training script configures Panda3D to use p3headlessgl (EGL) which
# provides GPU-accelerated headless rendering without needing Xvfb.a


# on runpod need to instal dependencies:  apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglvnd0 libx11-6 libxext6 libsm6 libice6
# and set:    export NVIDIA_VISIBLE_DEVICES=all
   #export NVIDIA_DRIVER_CAPABILITIES=all
   #export __GLX_VENDOR_LIBRARY_NAME=nvidia
# =============================================================================

set -e  # Exit on error

# Enable all NVIDIA capabilities including graphics
# By default, RunPod only enables 'compute,utility' which breaks OpenGL rendering
export NVIDIA_DRIVER_CAPABILITIES=all

# Tell GLX to use the NVIDIA vendor library (fallback if EGL fails)
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Suppress Panda3D device errors for missing /dev/input on headless servers
export PANDA_PRC_DATA="notify-level-device error"

echo "=============================================================="
echo "MetaDrive Headless Training Wrapper"
echo "=============================================================="
echo "Environment configured for EGL headless GPU rendering:"
echo "  NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
echo "  Using EGL (p3headlessgl) for hardware-accelerated rendering"
echo "=============================================================="
echo ""

# Run directly with EGL - no Xvfb needed!
exec uv run python "$@"
