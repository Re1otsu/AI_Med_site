# frontend/doctor_panel.py
import os
import sys
import uuid
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ---------- локальные модули ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.db import (
    get_history,
    get_patient,
    init_db,
    insert_or_update_patient,
    list_patients,
)
from app.chat_local import local_ai_chat

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

import app.predictor as P
print("LOADED PREDICTOR FROM:", P.__file__)

# ---------- базовая настройка страницы ----------
st.set_page_config(
    page_title="HealHub – Панель врача",
    layout="wide",
    page_icon="🏥",
)

# ---------- CSS: сдержанный «официальный» стиль ----------
st.markdown(
    """
<style>
/* скрываем стандартные части стримлита */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body { background-color: #F9FAFB; }

/* верхняя панель */
.top-nav {
  position: sticky; top: 0; z-index: 50;
  background: white; padding: 12px 20px;
  border-bottom: 1px solid #E5E7EB;
  display: flex; align-items: center; justify-content: space-between;
}

.top-nav-left { display: flex; align-items: center; gap: 10px; }
.top-nav-logo {
  width: 30px; height: 30px; border-radius: 999px;
  background: linear-gradient(135deg, #2563EB, #0EA5E9);
  color: white; font-weight: 700; font-size: 16px;
  display: flex; align-items: center; justify-content: center;
}
.top-nav-title { font-size: 18px; font-weight: 700; color: #111827; }
.top-nav-subtitle { font-size: 12px; color: #6B7280; margin-top: 2px; }

/* метрики */
.metric-card {
  padding: 14px 16px; border-radius: 12px; background: white;
  border: 1px solid #E5E7EB;
}
.metric-label { font-size: 11px; text-transform: uppercase; color: #6B7280; letter-spacing: .05em; }
.metric-value { font-size: 22px; font-weight: 700; color: #111827; margin-top: 4px; }
.metric-extra { font-size: 12px; color: #6B7280; }

/* карточки */
.card {
  padding: 16px 18px; border-radius: 12px; background: white;
  border: 1px solid #E5E7EB; box-shadow: 0 4px 10px rgba(15,23,42,0.03);
}

/* бейджи риска */
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.badge-high { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; }
.badge-medium { background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; }
.badge-low { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; }

/* разделитель-блок (визуальная пауза) */
.section { margin-top: 10px; }

/* кнопка-акцент */
button[kind="primary"] { font-weight: 600; }

/* контейнер ассистента справа */
.assistant {
  position: relative;
  height: calc(100% - 0px);
}
.chat-box {
  max-height: 60vh; overflow: auto; padding-right: 4px;
  border: 1px solid #E5E7EB; border-radius: 10px; padding: 8px 10px; background: #FBFBFD;
}
.chat-msg { margin: 6px 0; padding: 10px 12px; border-radius: 10px; }
.chat-user { background: #F3F4F6; }
.chat-ai { background: #EEF6FF; }
.chat-label { font-size:12px; font-weight:600; margin-bottom:4px; color:#6B7280; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- верхняя панель ----------
st.markdown(
    """
<div class="top-nav">
  <div class="top-nav-left">
    <div class="top-nav-logo">AI</div>
    <div>
      <div class="top-nav-title">HealHub</div>
      <div class="top-nav-subtitle">Панель врача • ЭКГ / МРТ / ФЛГ</div>
    </div>
  </div>
  <div style="font-size:13px;color:#4B5563;"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- данные ----------
init_db()

from app.db import migrate_db
migrate_db()

all_patients = list_patients() or []

# метрики
total_patients = len(all_patients)
high_risk = sum(1 for p in all_patients if p.get("risk") == "high")
week_ago = datetime.now() - timedelta(days=7)
recent = 0
for p in all_patients:
    ts = p.get("created_at")
    if not ts: continue
    try:
        dt = datetime.fromisoformat(str(ts).split(".")[0])
        if dt >= week_ago: recent += 1
    except:  # noqa
        pass

# ---------- блок метрик ----------
m1, m2, m3 = st.columns([1, 1, 1])
with m1:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">Пациенты</div>
        <div class="metric-value">{total_patients}</div>
        <div class="metric-extra">в базе</div>
        </div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">Высокий риск</div>
        <div class="metric-value">{high_risk}</div>
        <div class="metric-extra">требуют внимания</div>
        </div>""",
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">Исследования</div>
        <div class="metric-value">{recent}</div>
        <div class="metric-extra">за 7 дней</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div class='section'></div>", unsafe_allow_html=True)

# ---------- основной двухколоночный макет ----------
left, right = st.columns([2.3, 1.2], gap="large")

# ===================== ЛЕВАЯ КОЛОНКА =====================
with left:
    tabs = st.tabs(["➕ Новый анализ", "📋 Очередь пациентов"])

    # -------- Новый анализ --------
    with tabs[0]:
        st.subheader("Новый анализ исследования")

        st.markdown("Заполните данные пациента и загрузите изображение для анализа.")

        # ---- поля формы (без form!) ----
        name = st.text_input(
            "ФИО пациента",
            placeholder="Иванов Иван Иванович",
            key="new_name"
        )

        modality_ui = st.selectbox(
            "Тип исследования",
            ["Автоопределение", "ЭКГ", "МРТ", "Флюорография"],
            index=0,
            key="new_mod"
        )

        # ---- состояние превью ----
        if "show_preview" not in st.session_state:
            st.session_state["show_preview"] = True

        # --- init ---
        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0
        if "show_preview" not in st.session_state:
            st.session_state.show_preview = True

        # --- uploader with dynamic key ---
        uploaded = st.file_uploader(
            "Изображение (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.upload_key}"
        )

        # --- preview ---
        pil_img = None
        if uploaded and st.session_state.show_preview:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Загруженный снимок", use_container_width=True)

        # --- analyze button ---
        analyze_clicked = st.button(
            "🔍 Проанализировать и сохранить",
            type="primary",
            disabled=not (name and pil_img),
        )

        if analyze_clicked:
            with st.spinner("Выполняется анализ снимка..."):
                from app import predictor

                # маппинг модальности
                forced_map = {
                    "Автоопределение": None,
                    "ЭКГ": "ecg",
                    "МРТ": "mri",
                    "Флюорография": "xray",
                }

                forced = forced_map.get(modality_ui)

                summary, heatmap_path, payload = predictor.predict_image(
                    pil_img,
                    STORAGE_DIR,
                    forced_modality=forced
                )


            # save original
            uid = uuid.uuid4().hex[:8]
            orig_path = os.path.join(STORAGE_DIR, f"{uid}_orig.png")
            pil_img.save(orig_path)

            # save heatmap
            hmap = None
            if heatmap_path and os.path.exists(heatmap_path):
                import shutil
                new_hm = os.path.join(STORAGE_DIR, f"{uid}_heatmap.png")
                shutil.copyfile(heatmap_path, new_hm)
                hmap = new_hm

            # save to database
            pid = insert_or_update_patient(
                name.strip(),
                payload,
                orig_path,
                hmap
            )

            st.success(f"Запись сохранена: пациент #{pid}")

            # --- hide preview ---
            st.session_state.show_preview = False

            # --- full reset uploader ---
            st.session_state.upload_key += 1

            st.rerun()


    # -------- Очередь пациентов --------
    with tabs[1]:
        st.subheader("Очередь пациентов")

        if not all_patients:
            st.info("Пока нет записей. Добавьте пациента во вкладке «Новый анализ».")
        else:
            risk_map = {"high":"🚨 Высокий","medium":"⚠️ Средний","low":"✅ Низкий"}
            mod_ru = {"ECG":"ЭКГ","MRI":"МРТ","X-ray":"Флюорография"}

            with st.expander("Фильтры", expanded=False):
                cfa, cfb, cfc = st.columns([1,1,1.2])
                with cfa:
                    risk_filter = st.multiselect(
                        "Риск", ["high","medium","low"], format_func=lambda r: risk_map.get(r,r)
                    )
                with cfb:
                    mod_filter = st.multiselect(
                        "Тип", ["ECG","MRI","X-ray"], format_func=lambda m: mod_ru.get(m,m)
                    )
                with cfc:
                    name_filter = st.text_input("Поиск по ФИО", placeholder="Начните вводить фамилию")

            filtered = []
            for p in all_patients:
                if risk_filter and p.get("risk") not in risk_filter: continue
                if mod_filter and p.get("modality") not in mod_filter: continue
                if name_filter and name_filter.lower() not in (p.get("name") or "").lower(): continue
                filtered.append(p)

            if not filtered:
                st.warning("По выбранным фильтрам пациенты не найдены.")
            else:
                df = pd.DataFrame(filtered)[
                    ["id","name","modality","label","risk","probability","created_at"]
                ].rename(columns={
                    "id":"№","name":"ФИО","modality":"Тип","label":"Заключение",
                    "risk":"Риск","probability":"Вероятность, %","created_at":"Создано"
                })
                df["Риск"] = df["Риск"].map(lambda r: risk_map.get(r, r))
                df["Тип"] = df["Тип"].replace(mod_ru)

                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "№": st.column_config.NumberColumn(width="small"),
                        "Вероятность, %": st.column_config.NumberColumn(format="%.2f", width="small"),
                        "Создано": st.column_config.TextColumn(width="medium"),
                    }
                )

                # выбор пациента
                id_to_patient = {p["id"]: p for p in filtered}
                selected_pid = st.selectbox(
                    "Карточка пациента:",
                    options=list(id_to_patient.keys()),
                    format_func=lambda pid: f"#{pid} — {id_to_patient[pid]['name']} — {id_to_patient[pid]['label']}",
                )

                if st.button("Открыть карточку", type="primary"):
                    p = get_patient(int(selected_pid))
                    if not p:
                        st.warning("Пациент не найден.")
                    else:
                        st.markdown("")  # небольшая пауза
                        c1, c2 = st.columns([1.4, 1.4])

                        with c1:
                            st.markdown(
                                f"""<div class="card">
                                <h4 style="margin:0;">🧾 Пациент #{p['id']}: {p['name']}</h4>
                                <p style="font-size:13px;color:#6B7280;margin:.4rem 0 0;">
                                  Дата записи: {p.get('created_at') or '—'}
                                </p></div>""",
                                unsafe_allow_html=True,
                            )
                            if p.get("image_path") and os.path.exists(p["image_path"]):
                                st.image(p["image_path"], caption="Исходное изображение", use_container_width=True)
                            else:
                                st.info("Исходное изображение не найдено.")
                            if p.get("heatmap_path") and os.path.exists(p["heatmap_path"]):
                                st.image(p["heatmap_path"], caption="Тепловая карта (Grad‑CAM)", use_container_width=True)
                            else:
                                st.caption("Тепловая карта не сохранена или недоступна.")

                        with c2:
                            st.markdown("#### Клиническая сводка")
                            risk = (p.get("risk") or "low")
                            risk_badge = {
                                "high": '<span class="badge badge-high">Высокий риск</span>',
                                "medium": '<span class="badge badge-medium">Средний риск</span>',
                                "low": '<span class="badge badge-low">Низкий риск</span>',
                            }.get(risk, '<span class="badge">—</span>')

                            st.markdown(f"**Тип исследования:** {p.get('modality') or '—'}")
                            st.markdown(f"**Заключение (модель):** {p.get('label') or '—'}")
                            st.markdown(f"**Вероятность:** {p.get('probability') or '—'}%")
                            st.markdown(f"**Риск:** {risk_badge}", unsafe_allow_html=True)
                            st.markdown(f"**Комментарий ИИ:** {p.get('diagnosis') or '—'}")
                            st.caption("Система носит рекомендательный характер и не заменяет врача.")

                        import altair as alt

                        st.markdown("#### 📈 Динамика пациента")

                        hist = get_history(int(selected_pid))
                        if not hist:
                            st.info("История для этого пациента пока пуста.")
                        else:
                            hdf = pd.DataFrame(hist).copy()

                            # аккуратно парсим время
                            if "timestamp" in hdf.columns:
                                hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], errors="coerce")
                                hdf = hdf.dropna(subset=["timestamp"]).sort_values("timestamp")
                            else:
                                st.info("В истории нет поля времени, графики построить нельзя.")
                                st.stop()

                            # если modality ещё не было (старые записи) — ставим Unknown
                            if "modality" not in hdf.columns:
                                hdf["modality"] = "Unknown"

                            # словари для подписи
                            mod_ru = {"ECG": "ЭКГ", "MRI": "МРТ", "X-ray": "Флюорография", "Unknown": "Без типа"}
                            risk_ru = {"low": "Низкий", "medium": "Средний", "high": "Высокий"}

                            # ---------- общая таблица всех исследований ----------
                            st.markdown("#### 📋 Все исследования пациента")

                            label_ru_map = {
                                "Normal": "Норма",
                                "Arrhythmia": "Аритмия",
                                "Critical": "Критическое состояние",
                                "glioma": "Глиома",
                                "meningioma": "Менингиома",
                                "pituitary": "Опухоль гипофиза",
                                "notumor": "Без признаков опухоли",
                                "🟢 Вероятно норма": "🟢 Вероятно норма",
                                "🟡 Подозрительно": "🟡 Подозрительно",
                                "🔴 Критично": "🔴Критично",
                            }
                           
                            table_df = hdf[["timestamp", "modality", "label", "probability", "risk"]].copy()
                            table_df["label"] = table_df["label"].map(label_ru_map).fillna(table_df["label"])
                            table_df["modality"] = table_df["modality"].map(mod_ru).fillna(table_df["modality"])
                            table_df["risk"] = table_df["risk"].map(risk_ru).fillna(table_df["risk"])
                            table_df = table_df.rename(
                                columns={
                                    "timestamp": "Время",
                                    "modality": "Тип исследования",
                                    "label": "Заключение",
                                    "probability": "Вероятность, %",
                                    "risk": "Риск",
                                }
                            )
                            st.dataframe(table_df, use_container_width=True, hide_index=True)

                            # ---------- вкладки по типам исследований ----------
                            st.markdown("#### 🔍 Динамика по типам исследований")

                            # порядок типов
                            mods_order = ["ECG", "MRI", "X-ray", "Unknown"]
                            mods_in_data = [m for m in mods_order if m in set(hdf["modality"])]
                            if not mods_in_data:
                                mods_in_data = sorted(hdf["modality"].dropna().unique().tolist())

                            tabs = st.tabs(
                                [f"{mod_ru.get(m, m)} ({(hdf['modality'] == m).sum()})" for m in mods_in_data]
                            )

                            risk_domain = ["low", "medium", "high"]
                            risk_range = ["#10B981", "#F59E0B", "#EF4444"]  # зелёный / янтарный / красный

                            for tab, mod in zip(tabs, mods_in_data):
                                with tab:
                                    df_mod = hdf[hdf["modality"] == mod].copy()
                                    if df_mod.empty:
                                        st.info("Для этого типа нет исследований.")
                                        continue

                                    st.markdown(f"##### {mod_ru.get(mod, mod)}")

                                    # 1) Мини-метрика Health Index
                                    from app.db import health_index
                                    df_mod["health"] = df_mod.apply(lambda r: health_index(r["label"], r["risk"]), axis=1)

                                    if len(df_mod) >= 2:
                                        delta_h = df_mod["health"].iloc[-1] - df_mod["health"].iloc[0]
                                        current_h = df_mod["health"].iloc[-1]
                                        st.metric(
                                            "Индекс состояния здоровья",
                                            f"{current_h:.0f}/100",
                                            f"{delta_h:+.0f} пунктов"
                                        )
                                    else:
                                        st.metric(
                                            "Индекс состояния здоровья",
                                            f"{df_mod['health'].iloc[-1]:.0f}/100",
                                            "только одно измерение"
                                        )

                                    # 2) ГРАФИК Health Index
                                    chart_health = (
                                        alt.Chart(df_mod)
                                        .mark_line(point=True)
                                        .encode(
                                            x=alt.X("timestamp:T", title="Дата/время"),
                                            y=alt.Y("health:Q", title="Индекс здоровья (0–100)", scale=alt.Scale(domain=[0, 100])),
                                            tooltip=[
                                                alt.Tooltip("timestamp:T", title="Время"),
                                                alt.Tooltip("label:N", title="Заключение"),
                                                alt.Tooltip("risk:N", title="Риск"),
                                                alt.Tooltip("health:Q", title="Health Index"),
                                            ],
                                            color=alt.value("#2563EB")
                                        )
                                        .properties(height=240)
                                    )

                                    st.altair_chart(chart_health, use_container_width=True)

# ===================== ПРАВАЯ КОЛОНКА (Ассистент) =====================
with right:
    st.markdown("### 🧠 Ассистент")
    # статус Ollama
    try:
        requests.get("http://localhost:11434", timeout=2)
        ollama_online = True
        st.caption("🟢 Ollama запущен локально")
    except Exception:
        ollama_online = False
        st.caption("🔴 Ollama недоступен. Запустите `ollama serve`.")

    model_name = st.selectbox("Модель ИИ", ["llama3", "phi3"], index=0, key="assistant_model")

    # история чата (в сессии)
    if "chat_global" not in st.session_state:
        st.session_state["chat_global"] = [
            {"role":"assistant","text":"Здравствуйте. Готов помочь по пациентам и исследованиям."}
        ]

    st.markdown('<div class="assistant">', unsafe_allow_html=True)
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)

    # выводим историю
    for msg in st.session_state["chat_global"]:
        css = "chat-ai" if msg["role"] == "assistant" else "chat-user"
        who = "Ассистент" if msg["role"] == "assistant" else "Врач"
        st.markdown(f'<div class="chat-msg {css}"><div class="chat-label">{who}</div>{msg["text"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # локальное поле ввода (чтобы не прыгало вниз страницы)
    with st.form("assistant_form", clear_on_submit=True):
        q = st.text_area("Сообщение ассистенту:", height=80, placeholder="Кратко опишите вопрос или просьбу…")
        send = st.form_submit_button("Отправить", type="primary")

    if send and q:
        st.session_state["chat_global"].append({"role":"user","text":q})

        # контекст пациентов
        pts = list_patients()
        if pts:
            ctx = "Текущие пациенты:\n" + "\n".join(
                f"- {p['name']}: {p['modality']} → {p['label']} (риск {p['risk']}, {p['probability']}%)"
                for p in pts
            )
        else:
            ctx = "Пациентов в базе нет."

        if not ollama_online:
            st.session_state["chat_global"].append(
                {"role":"assistant","text":"⚠️ Локальная модель Ollama не запущена."}
            )
        else:
            prompt = f"""Вы — медицинский ассистент.
Используйте данные ниже, отвечайте кратко, по-русски, без домыслов.

{ctx}

Вопрос: {q}"""
            ans = local_ai_chat(prompt, model=model_name)
            st.session_state["chat_global"].append({"role":"assistant","text": ans or "Ответ не получен."})

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- футер ----------
st.markdown(
    """
<hr style="margin-top:18px;margin-bottom:6px;">
<div style="font-size:12px;color:#9CA3AF;">
  © 2025 AI CardioCare. Не является медицинским изделием. Для принятия клинических решений требуется консультация врача.
</div>
""",
    unsafe_allow_html=True,
)
