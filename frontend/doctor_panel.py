# frontend/doctor_panel.py
import os, uuid, sys
import streamlit as st
from PIL import Image
import pandas as pd
import requests

# локальные модули
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.predictor import predict_image
from app.db import init_db, insert_or_update_patient, list_patients, get_patient, get_history
from app.chat_local import local_ai_chat

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

st.set_page_config(page_title="AI Doctor Panel", layout="wide")

st.markdown(
    """
    <h2 style='color:#2c3e50; font-weight:700;margin:0'>🏥 AI Doctor Panel</h2>
    <p style='color:#555;margin-top:6px'>Система анализа ЭКГ/МРТ/ФЛГ с подсветкой аномалий (Grad-CAM) и приоритезацией по риску.</p>
    """,
    unsafe_allow_html=True,
)

init_db()

view = st.sidebar.radio("Режим работы:", ["➕ Добавить/Проанализировать", "📋 Очередь пациентов"])

# -------- Добавление / анализ --------
if view == "➕ Добавить/Проанализировать":
    st.subheader("Добавление нового анализа")
    name = st.text_input("ФИО пациента", placeholder="Иванов Иван Иванович")
    uploaded = st.file_uploader("Изображение (JPG/PNG)", type=["jpg","jpeg","png"])

    if uploaded and name:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Загруженный снимок", use_container_width=True)

        if st.button("🔍 Проанализировать и сохранить"):
            with st.spinner("Выполняется анализ..."):
                from app import predictor  # локальные вызовы
                # Всегда просим сохранять Grad-CAM в storage:
                summary, heatmap_path, payload = predictor.predict_image(pil_img, workdir=STORAGE_DIR)

            # сохраняем исходник
            uid = uuid.uuid4().hex[:8]
            src_path = os.path.join(STORAGE_DIR, f"{uid}_orig.png")
            pil_img.save(src_path)

            # копируем Grad-CAM рядом (если есть)
            hmap = None
            if heatmap_path and os.path.exists(heatmap_path):
                import shutil
                new_hm = os.path.join(STORAGE_DIR, f"{uid}_heatmap.png")
                shutil.copyfile(heatmap_path, new_hm)
                hmap = new_hm

            pid = insert_or_update_patient(name.strip(), payload, src_path, hmap)
            st.success(f"✅ Сохранено как запись пациента #{pid}")
            st.markdown(f"**Резюме:** {summary}")
            st.json(payload)

# -------- Очередь --------
elif view == "📋 Очередь пациентов":
    st.subheader("Очередь пациентов (по уровню риска)")
    rows = list_patients()
    if not rows:
        st.info("Пока нет записей. Добавьте пациента во вкладке «Добавить/Проанализировать».")
    else:
        df = pd.DataFrame(rows)[["id","name","modality","label","risk","probability","created_at"]]
        df = df.rename(columns={
            "id":"№","name":"ФИО пациента","modality":"Тип","label":"Заключение",
            "risk":"Риск","probability":"Вероятность, %","created_at":"Создано"
        })
        risk_map = {"high":"🚨 Высокий","medium":"⚠️ Средний","low":"✅ Низкий"}
        df["Риск"] = df["Риск"].map(lambda r: risk_map.get(r, r))
        df["Тип"] = df["Тип"].replace({"ECG":"ЭКГ","MRI":"МРТ","X-ray":"Флюорография"})
        st.dataframe(df, use_container_width=True)

        pid = st.number_input("ID пациента для просмотра:", min_value=1, step=1)
        if st.button("Открыть карточку"):
            p = get_patient(int(pid))
            if not p:
                st.warning("Пациент не найден.")
            else:
                st.markdown("---")
                st.markdown(f"### 🧾 Карточка #{p['id']}: **{p['name']}**")
                c1, c2 = st.columns([2,2])
                with c1:
                    if p.get("image_path") and os.path.exists(p["image_path"]):
                        st.image(p["image_path"], caption="Исходное изображение", use_container_width=True)
                    if p.get("heatmap_path") and os.path.exists(p["heatmap_path"]):
                        st.image(p["heatmap_path"], caption="Тепловая карта (Grad-CAM)", use_container_width=True)
                with c2:
                    st.markdown(f"**Тип:** {p.get('modality')}")
                    st.markdown(f"**Заключение:** {p.get('label')}")
                    st.markdown(f"**Вероятность:** {p.get('probability')}%")
                    st.markdown(f"**Риск:** {p.get('risk')}")
                    st.markdown(f"**Комментарий ИИ:** {p.get('diagnosis')}")
                    st.caption(f"Дата: {p.get('created_at')}")

                st.markdown("#### 📈 История наблюдений")
                hist = get_history(int(pid))
                if hist:
                    hdf = pd.DataFrame(hist)[["timestamp","label","probability","risk"]]
                    st.dataframe(hdf, use_container_width=True)
                else:
                    st.info("История пуста.")

# -------- Глобальный чат --------
st.markdown("---")
st.markdown("### 🧠 Ассистент медицинской системы")

try:
    requests.get("http://localhost:11434", timeout=2)
    st.success("🟢 Ollama запущен")
    ollama_online = True
except Exception:
    st.warning("🔴 Ollama недоступен. Запустите `ollama serve`.")
    ollama_online = False

model_name = st.selectbox("Модель ИИ:", ["llama3","phi3"], index=0, key="global_model")

if "chat_global" not in st.session_state:
    st.session_state["chat_global"] = [
        {"role":"assistant","text":"Здравствуйте. Готов помочь по пациентам и исследованиям."}
    ]

for msg in st.session_state["chat_global"]:
    with st.chat_message("assistant" if msg["role"]=="assistant" else "user"):
        st.markdown(msg["text"])

if q := st.chat_input("Напишите вопрос ассистенту..."):
    st.session_state["chat_global"].append({"role":"user","text":q})
    # контекст из базы
    pts = list_patients()
    if pts:
        ctx = "Текущие пациенты:\n" + "\n".join(
            f"- {p['name']}: {p['modality']} → {p['label']} (риск {p['risk']}, {p['probability']}%)" for p in pts
        )
    else:
        ctx = "Пациентов в базе нет."

    if not ollama_online:
        st.session_state["chat_global"].append({"role":"assistant","text":"⚠️ Ollama не запущен."})
    else:
        prompt = f"""Вы — медицинский ассистент.
Используйте данные ниже, отвечайте кратко, по-русски, без домыслов.

{ctx}

Вопрос: {q}"""
        ans = local_ai_chat(prompt, model=model_name)
        st.session_state["chat_global"].append({"role":"assistant","text": ans or "Ответ не получен."})
    st.rerun()
