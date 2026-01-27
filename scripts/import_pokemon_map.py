"""
Import Pokemon world map data (CSV) into MySQL.

This is a dev/ops utility that works both locally and in Docker Compose.

Default CSV:
  resources/data/map_data/finsh_with_coords.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pymysql


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def _connect(host: str, port: int, user: str, password: str, database: str):
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        autocommit=False,
    )


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pokemon_locations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pokemon_region VARCHAR(255),
    real_location VARCHAR(255),
    pokemon_list TEXT,
    latitude DOUBLE,
    longitude DOUBLE,
    real_address VARCHAR(255),
    exact_match VARCHAR(10)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
"""


def main() -> int:
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Import pokemon map CSV into MySQL.")
    parser.add_argument(
        "--csv",
        default=str(_repo_root() / "resources" / "data" / "map_data" / "finsh_with_coords.csv"),
        help="Path to finsh_with_coords.csv",
    )
    parser.add_argument("--force", action="store_true", help="Recreate table and re-import even if data exists.")
    parser.add_argument("--batch", type=int, default=500, help="Insert batch size (default: 500).")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    host = _env("MYSQL_HOST", "mysql_host", default="127.0.0.1")
    port = int(_env("MYSQL_PORT", "mysql_port", default="3307") or 3307)
    user = _env("MYSQL_USER", "mysql_user", default="root")
    password = _env("MYSQL_PASSWORD", "mysql_password", default="")
    database = _env("MYSQL_DATABASE", "mysql_database", default="langgraph")

    print(f"[import_map] csv={csv_path}")
    print(f"[import_map] mysql={host}:{port} db={database} user={user}")

    conn = _connect(host, port, user, password, database)
    try:
        with conn.cursor() as cur:
            if args.force:
                cur.execute("DROP TABLE IF EXISTS pokemon_locations")
                conn.commit()

            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

            cur.execute("SELECT COUNT(*) FROM pokemon_locations")
            existing = int(cur.fetchone()[0] or 0)
            if existing > 0 and not args.force:
                print(f"[import_map] pokemon_locations already has {existing} rows, skip (use --force to reimport).")
                return 0

            rows: list[tuple] = []
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(
                        (
                            row.get("宝可梦地区") or None,
                            row.get("现实地区") or None,
                            row.get("pokemon") or None,
                            float(row["纬度"]) if row.get("纬度") else None,
                            float(row["经度"]) if row.get("经度") else None,
                            row.get("推荐地址") or None,
                            row.get("是否精确匹配") or None,
                        )
                    )

            insert_sql = """
INSERT INTO pokemon_locations (
    pokemon_region, real_location, pokemon_list, latitude, longitude, real_address, exact_match
) VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

            cur.execute("TRUNCATE TABLE pokemon_locations")
            conn.commit()

            total = len(rows)
            for i in range(0, total, args.batch):
                chunk = rows[i : i + args.batch]
                cur.executemany(insert_sql, chunk)
                conn.commit()
                print(f"[import_map] {min(i + args.batch, total)}/{total}")

        print("[import_map] done.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
