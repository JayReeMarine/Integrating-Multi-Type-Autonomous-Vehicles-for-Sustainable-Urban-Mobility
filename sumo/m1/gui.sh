#!/bin/zsh
# Open the M1 scenario in sumo-gui (needs XQuartz).  Usage:  sumo/m1/gui.sh
# Playback starts automatically at 50 ms per simulated second; use the
# toolbar to pause, and the "Delay" box to slow down.
set -e
cd "$(dirname "$0")"
pgrep -q Xquartz || { echo "starting XQuartz..."; open -a XQuartz; sleep 4; }
export DISPLAY="${DISPLAY:-:0}"

# XQuartz registers its MIT-MAGIC-COOKIE under the hostname at the time it
# started; on a different network (new DHCP hostname) the lookup fails with
# "Authorization required".  Re-register the same cookie for the current name.
XAUTH=/opt/X11/bin/xauth
if [ -x "$XAUTH" ]; then
  cookie=$($XAUTH list 2>/dev/null | awk '/MIT-MAGIC-COOKIE-1/ {print $3; exit}')
  if [ -n "$cookie" ]; then
    $XAUTH add "$(hostname)/unix:0" MIT-MAGIC-COOKIE-1 "$cookie" 2>/dev/null || true
    $XAUTH add "$(hostname -s)/unix:0" MIT-MAGIC-COOKIE-1 "$cookie" 2>/dev/null || true
  fi
fi

exec ../../venv/bin/sumo-gui -c m1.sumocfg --start --delay 50 --window-size 1400,900 \
     --no-warnings --fcd-output /dev/null --tripinfo-output /dev/null
