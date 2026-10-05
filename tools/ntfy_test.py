"""
Send test pushes through the same code the drone uses (missions/notify.py), so
you can check your phone setup without flying.

    python tools/ntfy_test.py --url https://ntfy.sh/<your-topic>
    python tools/ntfy_test.py --url https://ntfy.sh/<your-topic> --photo some.jpg
    python tools/ntfy_test.py --url https://ntfy.sh/<your-topic> --launch-button \
        --phone-url http://drone.local:8080 --phone-token "$DRONE_TOKEN"
        # the button POSTs {"launch": true} to that phone page. Test it with main.py running
        # in a dry run (--driver print), never with a real drone on the bench, props on.

Sends: 1) an intrusion alert (with your photo, or a generated test picture),
2) an "object missing" alert (normal priority), 3) with --launch-button, a
sensor alert carrying the Launch + Open buttons. Prints what was sent.
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from missions.entities import Alert                     # noqa: E402
from missions.notify import AlertNotifier               # noqa: E402
from missions.responder import launch_actions           # noqa: E402


def test_picture(path: str) -> str:
    import cv2
    import numpy as np
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.rectangle(img, (280, 150), (360, 400), (0, 0, 255), 3)
    cv2.putText(img, "drone_follow test alert", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(path, img)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="test pushes via missions/notify.py")
    p.add_argument("--url", required=True, help="your ntfy topic URL, e.g. https://ntfy.sh/<your-unique-topic>")
    p.add_argument("--token", help="ntfy access token (self-hosted server with auth)")
    p.add_argument("--photo", help="attach this JPEG (default: a generated test picture)")
    p.add_argument("--launch-button", action="store_true")
    p.add_argument("--phone-url", default="http://drone.local:8080")
    p.add_argument("--phone-token")
    a = p.parse_args(argv)
    n = AlertNotifier(a.url, token=a.token, threaded=False, drone_id="test-drone")
    photo = a.photo or test_picture(os.path.join(tempfile.mkdtemp(), "test_alert.jpg"))
    n.submit(Alert("back gate", 0.0, 1.0, 12.0, 1, snapshot=photo, kind="intrusion", track="T1"))
    n.submit(Alert("study", 0.0, -5.0, 3.0, 0, kind="object_missing", detail="laptop"))
    if a.launch_button:
        if not a.phone_token:
            p.error("--launch-button needs --phone-token (the drone's --token)")
        n.submit(Alert("hall", 0.0, 6.0, -3.0, 0, kind="sensor", detail="pir-hall",
                       actions=launch_actions(a.phone_url, a.phone_token)))
    n.drain()
    st = n.stats()
    print(f"sent {st['sent']}, failed {st['failed']}" + (f", last error: {st['last_error']}" if st["last_error"] else ""))
    if st["failed"]:
        print("Check: the URL (https://<server>/<topic>), the token, and that this machine has internet.")
    return 0 if not st["failed"] else 1


if __name__ == "__main__":
    time.sleep(0)
    sys.exit(main())
