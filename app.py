import re
from pathlib import Path
import pandas as pd
import streamlit as st
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "prodotti.csv"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"

st.set_page_config(page_title="Assistente MGFishing", page_icon="🎣", layout="wide")

st.title("🎣 Assistente MGFishing")
st.write(
    "Benvenuto. Chiedimi pure informazioni sui prodotti, sulle spedizioni, "
    "sul tracking, sui resi, sull'assistenza e anche consigli tecnici di pesca."
)

API_KEY = st.secrets["OPENAI_API_KEY"]


def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def row_to_text(row):
    parts = []
    for col in row.index:
        val = safe_str(row[col]).strip()
        if val:
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def search_relevant_rows(df, question, top_n=20):
    words = [w.strip().lower() for w in question.split() if w.strip()]
    results = []

    for idx, row in df.iterrows():
        text = row_to_text(row).lower()
        score = 0
        for word in words:
            if word in text:
                score += 1
        if score > 0:
            results.append((score, idx, row))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        fallback = []
        for idx, row in df.head(top_n).iterrows():
            fallback.append((0, idx, row))
        return fallback

    return results[:top_n]


def load_knowledge_file():
    if KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def make_links_clickable(text):
    if not text:
        return text

    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)",
        r'<a href="\2" target="_blank">\1</a>',
        text,
    )

    text = re.sub(
        r"(https://wa\.me/\d+)",
        r'<a href="\1" target="_blank">\1</a>',
        text,
    )

    return text


def is_product_question(question):
    q = question.lower()
    product_keywords = [
        "prodotto", "prodotti", "mulinello", "mulinelli", "canna", "canne", "filo", "trecciato",
        "articolo", "articoli", "categoria", "prezzo", "marca", "disponibile", "disponibilità",
        "codice", "sku", "shimano", "daiwa", "trabucco", "colmic", "rapture", "yuki", "molix",
        "catalogo"
    ]
    return any(keyword in q for keyword in product_keywords)


def is_store_info_question(question):
    q = question.lower()
    store_keywords = [
        "spedizione", "spedizioni", "tracking", "reso", "resi", "pagamento", "pagamenti",
        "whatsapp", "assistenza", "contatti", "contatto", "ordine", "ordini", "corriere",
        "consegna", "consegne", "tempo di spedizione", "tempi di spedizione", "costo spedizione",
        "costi di spedizione", "poste italiane", "sda", "bartolini", "tnt", "fedex", "ups",
        "numero", "telefono", "supporto"
    ]
    return any(keyword in q for keyword in store_keywords)


def is_fishing_advice_question(question):
    q = question.lower()
    advice_keywords = [
        "montatura", "montature", "finale", "finali", "trave", "bracciolo", "terminale", "terminali",
        "spot", "spiaggia", "foce", "porto", "scogliera", "lago", "torrente", "fiume",
        "tecnica", "tecniche", "esca", "esche", "artificiale", "artificiali",
        "come pescare", "come prendere", "come insidiare", "preda", "pesce",
        "orata", "spigola", "serra", "cefalo", "trota", "carpa", "palamita", "alletterato",
        "mare mosso", "acqua torbida", "acqua velata", "vento", "stagione", "marea",
        "consiglio", "consigli", "assetto", "diametro", "amo", "ami", "galleggiante",
        "bombarda", "feeder", "surfcasting", "spinning", "bolognese", "trout area"
    ]
    return any(keyword in q for keyword in advice_keywords)


if "messages" not in st.session_state:
    st.session_state.messages = []

if not CSV_FILE.exists():
    st.error("File prodotti.csv non trovato nella cartella del progetto.")
    st.stop()

knowledge_text = load_knowledge_file()
if not knowledge_text:
    st.warning("knowledge.txt non trovato o vuoto.")

try:
    df = pd.read_csv(CSV_FILE, sep=";", engine="python")
except Exception as e:
    st.error(f"Errore nella lettura di prodotti.csv: {e}")
    st.stop()

client = OpenAI(api_key=API_KEY)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            rendered = make_links_clickable(msg["content"])
            st.markdown(rendered, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

question = st.chat_input("Scrivi qui la tua domanda")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    use_products = is_product_question(question)
    use_store_info = is_store_info_question(question)
    use_fishing_advice = is_fishing_advice_question(question)

    if not use_products and not use_store_info and not use_fishing_advice:
        use_products = True
        use_store_info = True
        use_fishing_advice = True

    context_text = ""

    if use_products:
        relevant_rows = search_relevant_rows(df, question, top_n=20)
        context_lines = []
        for score, idx, row in relevant_rows:
            context_lines.append(f"Riga {idx + 1}: {row_to_text(row)}")
        product_context = "\n".join(context_lines)
        context_text += f"CONTESTO PRODOTTI (CSV):\n{product_context}\n\n"

    if use_store_info and knowledge_text:
        context_text += f"CONTESTO NEGOZIO (KNOWLEDGE.TXT):\n{knowledge_text}\n\n"

    system_prompt = (
        "Sei l'assistente di MGFishing. "
        "Per prodotti, prezzi, disponibilità e catalogo usa solo il CSV. "
        "Per spedizioni, tracking, resi, contatti e assistenza usa solo knowledge.txt. "
        "Per consigli tecnici di pesca, montature, spot, tecniche, stagionalità ed esche puoi usare la ricerca web. "
        "Non mostrare mai fonti, siti, riferimenti o link esterni al cliente. "
        "Rispondi sempre in italiano, in modo chiaro, concreto e naturale. "
        "Non inventare dati del negozio o del catalogo. "
        "Se è presente un link WhatsApp del negozio, mantienilo cliccabile."
    )

    user_prompt = f"""
Domanda utente:
{question}

Contesto disponibile:
{context_text}

Istruzioni:
- Usa il CSV solo per i prodotti.
- Usa knowledge.txt solo per le informazioni del negozio.
- Se la domanda è tecnica di pesca, puoi usare la web search.
- Non mostrare riferimenti esterni.
- Se consigli prodotti, proponi solo prodotti coerenti con il CSV.
- Se mancano dati interni del negozio, dillo chiaramente.
"""

    tools = []
    if use_fishing_advice:
        tools.append({"type": "web_search_preview"})

    with st.chat_message("assistant"):
        with st.spinner("Sto leggendo i dati..."):
            response = client.responses.create(
                model="gpt-5.4",
                tools=tools,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.output_text
            rendered = make_links_clickable(answer)
            st.markdown(rendered, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:12px; color:gray;'>Powered By EMANUELE CENSORI</div>",
    unsafe_allow_html=True,
)
