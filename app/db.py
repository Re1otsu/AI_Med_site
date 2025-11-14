import os, sqlite3, datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "patients.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        modality TEXT,
        label TEXT,
        diagnosis TEXT,
        probability REAL,
        risk TEXT,
        created_at TEXT,
        image_path TEXT,
        heatmap_path TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        timestamp TEXT,
        modality TEXT,               -- ← ЭТОГО НЕ ХВАТАЕТ!
        label TEXT,
        diagnosis TEXT,
        probability REAL,
        risk TEXT,
        image_path TEXT,
        heatmap_path TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(id)
    )""")

    conn.commit()
    conn.close()

def risk_score(risk: str) -> int:
    return {"high":2,"medium":1,"low":0}.get(risk, 0)

def insert_or_update_patient(name: str, payload: Dict[str, Any], image_path: str, heatmap_path: Optional[str]) -> int:
    """
    If patient exists -> update main record and append to history.
    If new -> create and also create first history row.
    Returns patient_id.
    """
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    # compute risk tag
    risk = infer_risk(payload)

    # find patient
    cur.execute("SELECT id FROM patients WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        pid = row["id"]
        # update current snapshot
        cur.execute("""UPDATE patients SET modality=?, label=?, diagnosis=?, probability=?, risk=?, created_at=?, image_path=?, heatmap_path=?
                       WHERE id=?""",
                    (payload.get("modality"), payload.get("label"), payload.get("diagnosis"),
                     float(payload.get("probability",0.0)), risk, now, image_path, heatmap_path, pid))
    else:
        cur.execute("""INSERT INTO patients (name, modality, label, diagnosis, probability, risk, created_at, image_path, heatmap_path)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (name, payload.get("modality"), payload.get("label"), payload.get("diagnosis"),
                     float(payload.get("probability",0.0)), risk, now, image_path, heatmap_path))
        pid = cur.lastrowid

    # append to history
    cur.execute("""
    INSERT INTO history 
    (patient_id, timestamp, modality, label, diagnosis, probability, risk, image_path, heatmap_path)
    VALUES (?,?,?,?,?,?,?,?,?)""",
    (
        pid,
        now,
        payload.get("modality"),     # 🟢 добавили!
        payload.get("label"),
        payload.get("diagnosis"),
        float(payload.get("probability", 0.0)),
        risk,
        image_path,
        heatmap_path
    ))

    conn.commit()
    conn.close()
    return pid

def infer_risk(payload: Dict[str, Any]) -> str:
    # Normalize across modalities
    mod = (payload.get("modality") or "").lower()
    prob = float(payload.get("probability",0.0))
    label = (payload.get("label") or "").lower()

    if mod == "x-ray":
        return payload.get("risk_level","low")
    if mod == "ecg":
        if "critical" in label.lower():
            return "high"
        elif "arrhythmia" in label.lower():
            return "medium"
        else:
            return "low"
    if mod == "mri":
        if label in {"glioma","pituitary"} and prob >= 60:
            return "high"
        elif label in {"glioma","meningioma","pituitary"}:
            return "medium"
        else:
            return "low"
    return "low"

def list_patients() -> List[Dict[str,Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    # sort by risk score desc then probability desc
    rows.sort(key=lambda r: (risk_score(r.get("risk","low")), float(r.get("probability",0.0))), reverse=True)
    return rows

def get_patient(pid: int) -> Optional[Dict[str,Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id=?", (pid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_history(pid: int) -> list:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM history WHERE patient_id=? ORDER BY id DESC", (pid,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# --------------------------------------------
# Health Index (0–100) — общая шкала состояния пациента
# --------------------------------------------

def health_index(label: str, risk: str) -> int:
    """
    Преобразует диагноз и уровень риска в шкалу состояния 0–100.
    100 = полностью здоров
    0   = крайне тяжёлое состояние
    """

    label = (label or "").lower()
    risk = (risk or "low").lower()

    # базовые уровни по риску
    base = {
        "low": 85,
        "medium": 55,
        "high": 25
    }.get(risk, 70)

    # коррекция по диагнозу
    bad_keywords = [
        "critical", "infarkt", "stroke", "severe", "tumor",
        "glioma", "meningioma", "pituitary", "pneumonia"
    ]
    warn_keywords = [
        "arrhythmia", "block", "ischemia", "lesion", "nodule"
    ]

    # ухудшение при тяжелых диагнозах
    if any(w in label for w in bad_keywords):
        base -= 25
    elif any(w in label for w in warn_keywords):
        base -= 10

    # ограничение в пределах 0—100
    base = max(0, min(100, base))
    return int(base)

# ------------------------------------------------------
# 🔧 Автоматическая миграция структуры базы данных
# ------------------------------------------------------

def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    return column in cols


def migrate_db():
    """
    Автоматически обновляет структуру БД:
    - добавляет недостающие колонки
    - обеспечивает совместимость старых таблиц
    """
    conn = get_conn()
    cur = conn.cursor()

    # ---------- Проверяем таблицу history ----------
    cur.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    cur.execute("PRAGMA table_info(history)")
    existing_cols = [row[1] for row in cur.fetchall()]

    required = [
        ("patient_id", "INTEGER"),
        ("timestamp", "TEXT"),
        ("modality", "TEXT"),
        ("label", "TEXT"),
        ("diagnosis", "TEXT"),
        ("probability", "REAL"),
        ("risk", "TEXT"),
        ("image_path", "TEXT"),
        ("heatmap_path", "TEXT"),
    ]

    for col, col_type in required:
        if col not in existing_cols:
            print(f"[MIGRATION] Добавляю колонку history.{col}")
            cur.execute(f"ALTER TABLE history ADD COLUMN {col} {col_type}")

    # Заполняем modality, если пусто
    cur.execute("UPDATE history SET modality='Unknown' WHERE modality IS NULL OR modality=''")

    conn.commit()
    conn.close()
    print("✅ Миграция выполнена")
