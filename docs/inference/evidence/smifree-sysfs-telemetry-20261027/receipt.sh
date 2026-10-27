#!/bin/sh
# Side-by-side receipt: old aien-cli GPU telemetry (nvidia-smi) vs new (sysfs/procfs).
# Read-only. It reads sysfs/procfs, runs the old binary's exact nvidia-smi query as the
# oracle, runs `aien-cli doctor` (HTTP GETs to localhost 18006/18082/18080/18081/18085
# only) from both binaries, and writes only into OUT. It never writes sysfs, never
# touches 127.0.0.1:18091 beyond the /health GET, and loads nothing on the GPU.
#
# usage: receipt.sh OUT OLD_AIEN_CLI NEW_AIEN_CLI [N samples] [interval s]
set -eu
OUT=$1; OLD=$2; NEW=$3; N=${4:-30}; IVL=${5:-2}
mkdir -p "$OUT"

tgpu_mc() {
	for z in /sys/class/thermal/thermal_zone*; do
		if [ "$(cat "$z/device/path" 2>/dev/null)" = '\_TZ_.TGPU' ]; then cat "$z/temp"; return; fi
	done
	echo NA
}

# Pids holding an open fd on, or a mapping of, /dev/nvidia0|nvidiactl|nvidia-uvm
# (first-party replacement for --query-compute-apps pids). Only processes this
# user may inspect are seen; per-process GPU memory has no first-party source.
proc_pids() {
	for p in /proc/[0-9]*; do
		if ls -l "$p/fd" 2>/dev/null | grep -qE '/dev/nvidia(0|ctl|-uvm)$' ||
			grep -qE '/dev/nvidia(0|ctl|-uvm)' "$p/maps" 2>/dev/null; then
			echo "${p#/proc/}"
		fi
	done | sort -n | tr '\n' ' '
}

smi_pids() {
	nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | sort -n | tr '\n' ' '
}

{
	date -u +%FT%TZ
	echo "## health 18091 (before)"; curl -s -m 5 127.0.0.1:18091/health; echo
	echo "## kernel"; uname -r
	echo "## /proc/driver/nvidia/version"; head -1 /proc/driver/nvidia/version
	for d in /sys/bus/pci/devices/*; do
		c=$(cat "$d/class"); [ $((c >> 16)) -eq 3 ] || continue
		echo "## display device ${d##*/}"
		echo "vendor=$(cat "$d/vendor") device=$(cat "$d/device") class=$c"
		echo "driver=$(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null)"
		echo "power_state=$(cat "$d/power_state" 2>/dev/null) power/control=$(cat "$d/power/control" 2>/dev/null)"
		echo "hwmon=$(ls "$d/hwmon" 2>/dev/null | tr '\n' ' ')"
	done
	echo "## thermal zones (type, ACPI path, m°C)"
	for z in /sys/class/thermal/thermal_zone*; do
		echo "${z##*/} $(cat "$z/type") $(cat "$z/device/path" 2>/dev/null) $(cat "$z/temp")"
	done
	echo "## hwmon names"; for h in /sys/class/hwmon/hwmon*; do echo "${h##*/} $(cat "$h/name")"; done
	echo "## power_supply / powercap entries"; ls /sys/class/power_supply /sys/class/powercap 2>&1
} > "$OUT/system.txt" 2>&1

timeout 120 "$OLD" doctor > "$OUT/doctor-old.txt" 2>&1 || echo "rc=$?" >> "$OUT/doctor-old.txt"
timeout 120 "$NEW" doctor > "$OUT/doctor-new.txt" 2>&1 || echo "rc=$?" >> "$OUT/doctor-new.txt"

# ExecutionSurface::detect() (backend selection; printed by --version).
{ echo "old: $("$OLD" --version 2>&1)"; echo "new: $("$NEW" --version 2>&1)"; } > "$OUT/version.txt"
echo "utc,smi_name,smi_temp_c,smi_mem_used,smi_mem_total,smi_util_pct,tgpu_mc" > "$OUT/samples.csv"
i=0
while [ "$i" -lt "$N" ]; do
	smi=$(nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu \
		--format=csv,noheader,nounits 2>&1 | sed 's/, /,/g')
	echo "$(date -u +%FT%TZ),$smi,$(tgpu_mc)" >> "$OUT/samples.csv"
	i=$((i + 1)); sleep "$IVL"
done

# Interleaved GPU-user sets; order alternates so neither source always reads first.
echo "utc,first,smi_pids,proc_pids,equal" > "$OUT/gpu-users.csv"
i=0
while [ "$i" -lt "$N" ]; do
	if [ $((i % 2)) -eq 0 ]; then first=smi; s=$(smi_pids); q=$(proc_pids)
	else first=proc; q=$(proc_pids); s=$(smi_pids); fi
	[ "$s" = "$q" ] && eq=1 || eq=0
	echo "$(date -u +%FT%TZ),$first,$s,$q,$eq" >> "$OUT/gpu-users.csv"
	i=$((i + 1)); sleep "$IVL"
done

{ echo "## health 18091 (after)"; curl -s -m 5 127.0.0.1:18091/health; echo; date -u +%FT%TZ; } >> "$OUT/system.txt"
