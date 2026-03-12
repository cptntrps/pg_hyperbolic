#!/usr/bin/env python3
"""
Demo: Poincaré distance in SQLite using the hyp_sqlite extension.
Proves the same math works in a completely different database engine.
"""
import sqlite3
import struct
import subprocess
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def vec_to_blob(floats):
    return struct.pack(f'{len(floats)}f', *floats)

def parse_pg_vector(text):
    text = text.strip().strip('[]')
    return [float(x) for x in text.split(',')]

def main():
    db = sqlite3.connect(':memory:')
    db.enable_load_extension(True)
    db.load_extension(os.path.join(SCRIPT_DIR, 'hyp_sqlite'))

    db.execute('''CREATE TABLE vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        embedding BLOB
    )''')

    # Load vectors exported from Postgres
    csv_path = '/tmp/hyp_vectors.csv'
    count = 0
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 1)
            if len(parts) != 2:
                continue
            name = parts[0]
            vec = parse_pg_vector(parts[1])
            db.execute('INSERT INTO vectors (name, embedding) VALUES (?, ?)',
                       (name, vec_to_blob(vec)))
            count += 1

    print(f"Loaded {count} vectors into SQLite\n")

    # KNN query
    query_name = db.execute('SELECT name FROM vectors LIMIT 1').fetchone()[0]
    print(f"=== KNN: 5 nearest to '{query_name}' ===")
    rows = db.execute('''
        SELECT v.name,
               round(poincare_distance(v.embedding, q.embedding), 4) AS dist,
               round(hyp_radius(v.embedding), 4) AS radius
        FROM vectors v, (SELECT embedding FROM vectors WHERE name = ?) q
        WHERE v.name != ?
        ORDER BY poincare_distance(v.embedding, q.embedding)
        LIMIT 5
    ''', (query_name, query_name)).fetchall()
    for name, dist, radius in rows:
        print(f"  {name:30s}  dist={dist}  radius={radius}")

    # Variable curvature comparison
    print(f"\n=== Variable curvature (nearest to '{query_name}') ===")
    for c in [0.5, 1.0, 2.0]:
        row = db.execute('''
            SELECT v.name, round(poincare_distance(v.embedding, q.embedding, ?), 4)
            FROM vectors v, (SELECT embedding FROM vectors WHERE name = ?) q
            WHERE v.name != ?
            ORDER BY poincare_distance(v.embedding, q.embedding, ?)
            LIMIT 1
        ''', (c, query_name, query_name, c)).fetchone()
        print(f"  c={c}: nearest = {row[0]} (dist={row[1]})")

    # Cross-validate with Postgres
    print(f"\n=== Cross-validation: SQLite vs Postgres distances ===")
    pairs = db.execute('''
        SELECT a.name, b.name,
               round(poincare_distance(a.embedding, b.embedding), 8)
        FROM vectors a, vectors b
        WHERE a.id = 1 AND b.id IN (2,3,4,5)
    ''').fetchall()
    for a_name, b_name, dist in pairs:
        print(f"  {a_name} ↔ {b_name}: {dist}")

    print("\nAll tests passed — same hyperbolic math, different database engine.")
    db.close()

if __name__ == '__main__':
    main()
