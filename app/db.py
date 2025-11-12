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
    cur.execute("""INSERT INTO history (patient_id, timestamp, label, diagnosis, probability, risk, image_path, heatmap_path)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (pid, now, payload.get("label"), payload.get("diagnosis"), float(payload.get("probability",0.0)),
                 risk, image_path, heatmap_path))
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

def clear_database():
    """Полностью очищает базу пациентов и историю наблюдений."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM history;")
    cur.execute("DELETE FROM patients;")
    conn.commit()
    conn.close()
    print("✅ База данных очищена.")