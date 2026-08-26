#!/usr/bin/env python3
"""Display raw RGB frames from stdin in an isolated OpenCV process."""

import argparse
import sys


def read_frame(size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    args = parser.parse_args()

    import cv2
    import numpy as np

    frame_bytes = args.width * args.height * 3

    try:
        frame_index = 0
        while data := read_frame(frame_bytes):
            image_bgr = np.frombuffer(data, dtype=np.uint8).reshape(args.height, args.width, 3)[:, :, ::-1].copy()
            cv2.putText(image_bgr, f"dataset frame {frame_index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("BH05 D1 policy image", image_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
