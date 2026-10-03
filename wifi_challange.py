# Passive 802.11 beacon scanner: prints the SSID / BSSID / channel that nearby
# access points broadcast in the clear.
#
# Requires: pip install scapy, plus a wireless interface in monitor mode and
# root/sudo (Linux). Put the adapter in monitor mode first, e.g.:
#   sudo airmon-ng start wlan0        (creates wlan0mon)
import sys

from scapy.all import sniff, Dot11Beacon

IFACE = sys.argv[1] if len(sys.argv) > 1 else "wlan0mon"

seen = set()  # (bssid, ssid) pairs already printed, to avoid repeat beacons

def handle(pkt):
    if not pkt.haslayer(Dot11Beacon):
        return

    bssid = pkt.addr2  # transmitter address = the AP's MAC

    # network_stats() reads the beacon's information elements the robust way:
    # it locates the DS Parameter Set by its element ID rather than by position,
    # so the channel is still found when an AP sends Country/ERP/TIM tags first.
    stats = pkt[Dot11Beacon].network_stats()
    ssid = stats.get("ssid", "")            # empty string for hidden networks
    channel = stats.get("channel", "?")
    crypto = "/".join(sorted(stats.get("crypto", set()))) or "?"

    key = (bssid, ssid)
    if key in seen:
        return
    seen.add(key)

    print(f"SSID={ssid!r}  BSSID={bssid}  CH={channel}  SEC={crypto}")


print(f"Sniffing beacons on {IFACE!r} (Ctrl+C to stop)...")
try:
    sniff(iface=IFACE, prn=handle, store=False)
except PermissionError:
    sys.exit("Need root: run with sudo.")
except OSError as e:
    sys.exit(f"Could not open {IFACE!r}: {e}\nIs the interface in monitor mode?")
