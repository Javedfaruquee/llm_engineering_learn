"""Windows equivalent of wifi_challange.py.

The scapy version needs monitor mode, which Windows Wi-Fi drivers almost never
expose (WlanHelper reports the MediaTek MT7925 here as "managed" only). Instead
of sniffing raw beacons, this asks Windows for its own scan of nearby access
points via `netsh wlan show networks mode=bssid` and prints the same fields.

No admin rights, no Npcap, no monitor mode required. Run:
    python wifi_scan_windows.py
"""
import re
import subprocess
import sys


def scan():
    """Return netsh's BSSID scan as text, or exit with a clear message."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        sys.exit("netsh not found - this script only runs on Windows.")
    if result.returncode != 0:
        sys.exit(f"netsh failed: {result.stdout}{result.stderr}".strip()
                 + "\nIs the WLAN AutoConfig service running and Wi-Fi enabled?")
    return result.stdout


def parse(text):
    """Turn netsh output into one record per BSSID (radio) seen."""
    networks = []
    ssid = auth = enc = None
    current = None

    for line in text.splitlines():
        # A new SSID block resets the shared auth/encryption fields
        m = re.match(r"\s*SSID\s+\d+\s*:\s*(.*)", line)
        if m:
            ssid = m.group(1).strip()
            auth = enc = None
            continue

        m = re.match(r"\s*Authentication\s*:\s*(.*)", line)
        if m:
            auth = m.group(1).strip()
            continue

        m = re.match(r"\s*Encryption\s*:\s*(.*)", line)
        if m:
            enc = m.group(1).strip()
            continue

        # Each BSSID is one physical radio of that SSID; start a new record
        m = re.match(r"\s*BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})", line)
        if m:
            current = {
                "ssid": ssid, "bssid": m.group(1).lower(),
                "auth": auth, "enc": enc, "channel": "?", "signal": "?",
            }
            networks.append(current)
            continue

        if current is None:
            continue

        # Anchored to line start so the mid-line "Channel:" in Wi-Fi 7 MLO
        # sub-entries doesn't overwrite the real per-BSSID channel.
        m = re.match(r"\s*Channel\s*:\s*(\d+)", line)
        if m:
            current["channel"] = m.group(1)
            continue

        m = re.match(r"\s*Signal\s*:\s*(\d+)\s*%", line)
        if m:
            current["signal"] = m.group(1) + "%"

    return networks


def main():
    networks = parse(scan())
    if not networks:
        print("No networks visible. Turn Wi-Fi on and make sure a scan can run.")
        return

    # Strongest signal first
    networks.sort(key=lambda n: int(n["signal"].rstrip("%")) if n["signal"] != "?" else -1,
                  reverse=True)

    for n in networks:
        ssid = repr(n["ssid"]) if n["ssid"] else "<hidden>"
        sec = "/".join(p for p in (n["auth"], n["enc"]) if p) or "?"
        print(f"SSID={ssid}  BSSID={n['bssid']}  CH={n['channel']}  "
              f"SIG={n['signal']}  SEC={sec}")

    print(f"\n{len(networks)} radio(s) across "
          f"{len(set(n['ssid'] for n in networks))} network name(s).")


if __name__ == "__main__":
    main()
