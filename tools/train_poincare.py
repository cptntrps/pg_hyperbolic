#!/usr/bin/env python3
"""
Train Poincaré embeddings from a graph and load into PostgreSQL.

Usage:
  # From edge list CSV (source,target columns):
  python train_poincare.py --edges edges.csv --dim 32 --dsn "postgresql://user:pass@host/db" --table my_vectors

  # From PostgreSQL edge table directly:
  python train_poincare.py --edge-query "SELECT source, target FROM edges" --dim 32 --dsn "..." --table my_vectors

  # Just export to CSV (no database):
  python train_poincare.py --edges edges.csv --dim 32 --output embeddings.csv

Requirements:
  pip install gensim numpy psycopg2-binary
"""
import argparse
import csv
import sys
import numpy as np

def load_edges_csv(path):
    """Load edges from CSV file (source,target columns)."""
    edges = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                edges.append((row[0].strip(), row[1].strip()))
    return edges

def load_edges_sql(dsn, query):
    """Load edges from a PostgreSQL query."""
    import psycopg2
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute(query)
    edges = [(str(r[0]), str(r[1])) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return edges

def train_poincare(edges, dim=32, epochs=100, negative=10, burn_in=10):
    """Train Poincaré embeddings using Gensim."""
    from gensim.models.poincare import PoincareModel

    print(f"Training Poincaré embeddings: {len(edges)} edges, {dim}D, {epochs} epochs")
    model = PoincareModel(
        edges,
        size=dim,
        negative=negative,
        burn_in=burn_in,
    )
    model.train(epochs=epochs, print_every=max(1, epochs // 10))
    return model

def project_to_ball(vec, eps=1e-7):
    """Ensure vector is strictly inside the unit ball."""
    norm = np.linalg.norm(vec)
    max_norm = 1.0 - eps
    if norm >= max_norm:
        vec = vec * (max_norm / norm)
    return vec

def export_csv(model, output_path):
    """Export embeddings to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        dim = model.kv.vector_size
        writer.writerow(['name'] + [f'd{i}' for i in range(dim)])
        for node in model.kv.index_to_key:
            vec = project_to_ball(model.kv[node])
            writer.writerow([node] + [f'{v:.8f}' for v in vec])
    print(f"Exported {len(model.kv)} embeddings to {output_path}")

def load_to_postgres(model, dsn, table, column='poincare_embedding', name_column='name'):
    """Load embeddings into PostgreSQL as vector type."""
    import psycopg2

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    dim = model.kv.vector_size

    # Create table if not exists
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {name_column} TEXT PRIMARY KEY,
            {column} vector({dim})
        )
    """)

    # Upsert embeddings
    count = 0
    for node in model.kv.index_to_key:
        vec = project_to_ball(model.kv[node])
        vec_str = '[' + ','.join(f'{v:.8f}' for v in vec) + ']'
        cur.execute(f"""
            INSERT INTO {table} ({name_column}, {column})
            VALUES (%s, %s::vector({dim}))
            ON CONFLICT ({name_column}) DO UPDATE SET {column} = EXCLUDED.{column}
        """, (node, vec_str))
        count += 1

    conn.commit()
    print(f"Loaded {count} embeddings into {table}.{column}")

    # Suggest HNSW index
    print(f"\nTo create a Poincaré HNSW index:")
    print(f"  CREATE INDEX ON {table} USING hnsw ({column} vector_poincare_ops);")
    print(f"\nTo query nearest neighbors:")
    print(f"  SELECT * FROM {table} ORDER BY {column} <~> '[...]'::vector LIMIT 10;")

    cur.close()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Train Poincaré embeddings for pg_hyperbolic')
    parser.add_argument('--edges', help='CSV file with source,target columns')
    parser.add_argument('--edge-query', help='SQL query returning source,target pairs')
    parser.add_argument('--dsn', help='PostgreSQL connection string')
    parser.add_argument('--table', default='poincare_vectors', help='Target table name')
    parser.add_argument('--column', default='embedding', help='Vector column name')
    parser.add_argument('--dim', type=int, default=32, help='Embedding dimension (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--output', help='Export to CSV instead of database')
    args = parser.parse_args()

    # Load edges
    if args.edges:
        edges = load_edges_csv(args.edges)
    elif args.edge_query and args.dsn:
        edges = load_edges_sql(args.dsn, args.edge_query)
    else:
        parser.error('Provide --edges CSV or --edge-query with --dsn')

    if not edges:
        print("No edges found!", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(edges)} edges")
    nodes = set()
    for s, t in edges:
        nodes.add(s)
        nodes.add(t)
    print(f"Unique nodes: {len(nodes)}")

    # Train
    model = train_poincare(edges, dim=args.dim, epochs=args.epochs)

    # Output
    if args.output:
        export_csv(model, args.output)
    elif args.dsn:
        load_to_postgres(model, args.dsn, args.table, args.column)
    else:
        export_csv(model, 'poincare_embeddings.csv')

if __name__ == '__main__':
    main()
