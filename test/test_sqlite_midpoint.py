"""hyp_midpoint in the SQLite extension returns the hyperbolic midpoint (python3 test/test_sqlite_midpoint.py)."""
import math
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "sqlite" / "hyp_sqlite.c"


def blob(v):
    return struct.pack(f"{len(v)}f", *v)


def vec(b):
    return list(struct.unpack(f"{len(b) // 4}f", b))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        so = Path(tmp) / "hyp_sqlite.so"
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(SRC), "-lm"], check=True)
        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        db.load_extension(str(so))
        dist = lambda a, b: db.execute("SELECT poincare_distance(?, ?)", (blob(a), blob(b))).fetchone()[0]
        mid = lambda a, b: vec(db.execute("SELECT hyp_midpoint(?, ?)", (blob(a), blob(b))).fetchone()[0])
        fails = []
        a, b = [0.5, 0.0], [-0.2, 0.3]
        m = mid(a, b)
        if dist(mid([0.3, 0.1], [0.3, 0.1]), [0.3, 0.1]) > 1e-5:
            fails.append("midpoint of a point with itself is not the point")
        if abs(dist(a, m) - dist(a, b) / 2) > 1e-5 or abs(dist(m, b) - dist(a, b) / 2) > 1e-5:
            fails.append(f"midpoint {m} is not at half the distance from both ends")
        if dist(m, mid(b, a)) > 1e-5:
            fails.append("midpoint is not symmetric")
        a3, b3 = [0.9, 0.0, 0.1], [0.0, 0.9, -0.1]
        m3 = mid(a3, b3)
        if abs(dist(a3, m3) + dist(m3, b3) - dist(a3, b3)) > 1e-4 or abs(dist(a3, m3) - dist(m3, b3)) > 1e-4:
            fails.append("near the boundary the midpoint is off the geodesic")
        print("FAIL: " + "; ".join(fails) if fails else "PASS: sqlite hyp_midpoint")
        return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
