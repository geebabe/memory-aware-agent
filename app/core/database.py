import os
import time
import oracledb
from app.core.config import load_env

def setup_oracle_database(admin_user="system", admin_password="YourPassword123", dsn="127.0.0.1:1521/FREEPDB1",
                          vector_password="VectorPwd_2025"):
    """
    One-time admin setup: configures tablespace and VECTOR user.
    """
    print("=" * 60)
    print("ORACLE DATABASE SETUP")
    print("=" * 60)

    # Step 1: Connect as admin
    print("\n[1/4] Connecting as admin...")
    try:
        admin_conn = oracledb.connect(
            user=admin_user, password=admin_password, dsn=dsn
        )
        print(f"  Connected as {admin_user}")
    except Exception as e:
        print(f"  Admin connection failed: {e}")
        return False

    try:
        # Step 2: Find ASSM tablespace for JSON column support
        print("\n[2/4] Finding JSON-compatible (ASSM) tablespace...")
        assm_ts = _find_assm_tablespace(admin_conn)

        # Step 3: Create VECTOR user with ASSM default tablespace
        print("\n[3/4] Creating VECTOR user...")
        with admin_conn.cursor() as cur:
            ts_clause = f"DEFAULT TABLESPACE {assm_ts}" if assm_ts else ""
            cur.execute(f"""
                DECLARE
                    user_count NUMBER;
                BEGIN
                    SELECT COUNT(*) INTO user_count
                    FROM all_users WHERE username = 'VECTOR';
                    IF user_count = 0 THEN
                        EXECUTE IMMEDIATE
                            'CREATE USER VECTOR IDENTIFIED BY '
                            || '{vector_password} {ts_clause}';
                        EXECUTE IMMEDIATE
                            'GRANT CONNECT, RESOURCE, CREATE SESSION'
                            || ' TO VECTOR';
                        EXECUTE IMMEDIATE
                            'GRANT UNLIMITED TABLESPACE TO VECTOR';
                        EXECUTE IMMEDIATE
                            'GRANT CREATE TABLE, CREATE SEQUENCE,'
                            || ' CREATE VIEW TO VECTOR';
                    END IF;
                END;
            """)
            if assm_ts:
                cur.execute(f"ALTER USER VECTOR DEFAULT TABLESPACE {assm_ts}")
        admin_conn.commit()
        if assm_ts:
            print(f"  VECTOR user ready (default tablespace: {assm_ts})")
        else:
            print("  VECTOR user created but no ASSM tablespace found — JSON columns may fail (ORA-43853)")

    except Exception as e:
        print(f"  Warning during setup: {e}")
    finally:
        admin_conn.close()

    # Step 4: Test connection as VECTOR
    print("\n[4/4] Testing connection as VECTOR...")
    try:
        conn = oracledb.connect(
            user="VECTOR", password=vector_password, dsn=dsn
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM dual")
            cur.fetchone()
        conn.close()
        print("  Connection successful!")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    return True

def _find_assm_tablespace(conn):
    """Find an existing ASSM tablespace for JSON column support."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT TABLESPACE_NAME
                FROM USER_TABLESPACES
                WHERE SEGMENT_SPACE_MANAGEMENT = 'AUTO'
                  AND STATUS = 'ONLINE'
                ORDER BY CASE TABLESPACE_NAME
                    WHEN 'DATA' THEN 1
                    WHEN 'USERS' THEN 2
                    WHEN 'SYSAUX' THEN 3
                    ELSE 4
                END
            """)
            row = cur.fetchone()
            if row:
                print(f"  Found ASSM tablespace: {row[0]}")
                return row[0]
    except Exception as e:
        print(f"  USER_TABLESPACES query failed: {e}")

    # Step 2: No ASSM tablespace found — try creating DATA
    create_sqls = [
        "CREATE TABLESPACE DATA DATAFILE SIZE 500M AUTOEXTEND ON NEXT 100M MAXSIZE UNLIMITED SEGMENT SPACE MANAGEMENT AUTO"
    ]
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT FILE_NAME FROM DBA_DATA_FILES FETCH FIRST 1 ROW ONLY")
            row = cur.fetchone()
            if row:
                datafile_dir = os.path.dirname(row[0])
                create_sqls.insert(0, f"CREATE TABLESPACE DATA DATAFILE '{datafile_dir}/data01.dbf' SIZE 500M AUTOEXTEND ON NEXT 100M MAXSIZE UNLIMITED SEGMENT SPACE MANAGEMENT AUTO")
    except Exception:
        pass

    for sql in create_sqls:
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            print("  Created DATA tablespace (ASSM)")
            return 'DATA'
        except Exception as e:
            if "ORA-01543" in str(e):
                print("  DATA tablespace already exists")
                return 'DATA'
            continue

    print("  Could not find or create ASSM tablespace")
    return None

def connect_to_oracle(max_retries=3, retry_delay=5, user="system", password="YourPassword123",
                      dsn="127.0.0.1:1521/FREE", program="langchain_oracledb_deep_research_demo"):
    """Connect to Oracle database with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connection attempt {attempt}/{max_retries}...")
            conn = oracledb.connect(user=user, password=password, dsn=dsn, program=program)
            print("✓ Connected successfully!")
            return conn
        except oracledb.OperationalError as e:
            print(f"✗ Connection failed (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            raise
    raise ConnectionError("Failed to connect after all retries")

def table_exists(conn, table_name):
    """Check if a table exists in the current user's schema."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = UPPER(:table_name)", {"table_name": table_name})
        return cur.fetchone()[0] > 0

def safe_create_index(conn, vs, idx_name):
    """Create IVF vector index using raw SQL for maximum compatibility."""
    dist_map = {"COSINE": "COSINE", "EUCLIDEAN_DISTANCE": "EUCLIDEAN", "DOT_PRODUCT": "DOT"}
    dist = dist_map.get(vs.distance_strategy.name, "COSINE")
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE VECTOR INDEX {idx_name} ON {vs.table_name}(EMBEDDING) ORGANIZATION NEIGHBOR PARTITIONS DISTANCE {dist} WITH TARGET ACCURACY 95")
        print(f"  ✅ Created index: {idx_name}")
    except Exception as e:
        if "ORA-00955" in str(e):
            print(f"  ⏭️  Index already exists: {idx_name}")
        else:
            raise

def cleanup_vector_memory(conn, drop_tables: bool = False, table_prefix: str = None):
    """Clean up vector indexes and optionally tables."""
    dropped_indexes = 0
    dropped_tables = 0
    with conn.cursor() as cur:
        cur.execute("SELECT INDEX_NAME, TABLE_NAME FROM USER_INDEXES WHERE INDEX_TYPE = 'VECTOR' ORDER BY TABLE_NAME")
        indexes = cur.fetchall()
        for idx_name, table_name in indexes:
            if table_prefix and not table_name.upper().startswith(table_prefix.upper()):
                continue
            try:
                cur.execute(f"DROP INDEX {idx_name}")
                dropped_indexes += 1
            except Exception:
                pass
        if drop_tables:
            cur.execute("SELECT DISTINCT TABLE_NAME FROM USER_TAB_COLUMNS WHERE DATA_TYPE = 'VECTOR' ORDER BY TABLE_NAME")
            tables = cur.fetchall()
            for (table_name,) in tables:
                if table_prefix and not table_name.upper().startswith(table_prefix.upper()):
                    continue
                try:
                    cur.execute(f"DROP TABLE {table_name} PURGE")
                    dropped_tables += 1
                except Exception:
                    pass
        conn.commit()
    return {"indexes_dropped": dropped_indexes, "tables_dropped": dropped_tables}
