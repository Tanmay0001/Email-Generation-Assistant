import streamlit as st
import requests

st.set_page_config(page_title="Email Generation Assistant", page_icon="✉️")
st.title("✉️ Email Generation Assistant")
st.caption("Powered by Groq + LangChain")

col1, col2 = st.columns(2)
with col1:
    model = st.radio("Model", ["A (llama-3.3-70b)", "B (llama-3.1-8b)"], horizontal=True)
with col2:
    tone = st.selectbox("Tone", [
        "Professional and warm", "Urgent but respectful",
        "Apologetic and empathetic", "Humble and respectful",
        "Enthusiastic and persuasive", "Polite and friendly",
        "Confident and informative", "Assertive but collaborative",
        "Warm and welcoming", "Patient but persistent"
    ])

intent = st.text_input("Intent", placeholder="e.g. Follow up after a client meeting")
facts  = st.text_area("Key Facts (one per line, use - bullets)", height=120,
                       placeholder="- Client is Priya Mehta\n- Met on Monday\n- Next step is a demo")

if st.button("Generate Email", type="primary"):
    if not intent.strip():
        st.warning("Please enter an intent.")
    else:
        model_key = "A" if "A" in model else "B"
        with st.spinner("Generating..."):
            try:
                r = requests.post("http://127.0.0.1:8000/generate", json={
                    "intent": intent,
                    "facts": facts,
                    "tone": tone,
                    "model": model_key
                }, timeout=30)
                if r.ok:
                    data = r.json()
                    st.success("Email generated!")
                    st.text_area("Generated Email", data["email"], height=350)
                    st.caption(f"Model used: `{data['model_used']}`")
                else:
                    st.error(f"API error: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure uvicorn is running on port 8000.")