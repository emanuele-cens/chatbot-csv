import os
import re
import pandas as pd
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz

# =========================
# CONFIG
# =========================
WHATSAPP_NUMBER = "393494166335"
WHATSAPP_LABEL = "Emanuele"
WHATSAPP_URL = f"https://wa.me/{WHATSAPP_NUMBER}"

st.set_page_config(page_title="MGFishing Chatbot", page_icon="🎣", layout="centered")
st.title("🎣 MGFishing Chatbot")

# =========================
# OPENAI
# =========================
def get_openai_api_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

OPENAI_API_KEY = get_openai_api_key()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9àèéìòù\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_whatsapp_message(text: str = "") -> str:
    import urllib.parse
    base_msg = "Ciao Emanuele, vorrei informazioni su un prodotto MGFishing."
    if text:
        base_msg = f"Ciao Emanuele, vorrei informazioni su questo prodotto/richiesta: {text}"
    encoded = urllib.parse.quote(base_msg)
    return f"{WHATSAPP_URL}?text={encoded}"

def is_link_request(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "link", "mandami il link", "mi mandi il link", "dammi il link",
        "url", "pagina prodotto", "apri prodotto", "dove lo trovo"
    ]
    return any(k in t for k in keywords)

def is_product_request(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "mulinello", "canna", "filo", "trecciato", "monofilo", "fluorocarbon",
        "artificiale", "esca", "popper", "minnow", "bombarda", "galleggiante",
        "girella", "amo", "ami", "trabucco", "daiwa", "shimano", "colmic",
        "tubertini", "molix", "major craft", "rapture", "prodotto", "articolo",
        "misura", "taglia", "grammi", "azione", "frizione", "bobina"
    ]
    return any(k in t for k in keywords)

def is_general_info_request(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "meteo", "vento", "mare", "pioggia", "pressione", "temperatura",
        "montatura", "trave", "terminale", "surfcasting", "bolognese",
        "spinning", "trota", "feeder", "bolentino", "ledgering", "carpfishing",
        "come pescare", "come fare", "consiglio", "consigli", "tecnica"
    ]
    return any(k in t for k in keywords)

def fallback_whatsapp_reply(user_text: str = "") -> str:
    wa = make_whatsapp_message(user_text)
    return (
        f"Per questa richiesta ti consiglio di contattare direttamente **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
        f"[Scrivi su WhatsApp]({wa})"
    )

# =========================
# CSV LOADING
# =========================
@st.cache_data(show_spinner=False)
def load_catalog_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    original_columns = list(df.columns)
    lowered = {c.lower().strip(): c for c in original_columns}

    # Mappatura flessibile colonne
    possible_name_cols = ["name", "nome", "product_name", "titolo", "title", "prodotto", "descrizione_breve"]
    possible_url_cols = ["url", "link", "product_url", "permalink"]
    possible_desc_cols = ["description", "descrizione", "short_description", "descrizione_breve", "details"]
    possible_cat_cols = ["category", "categoria", "brand", "marchio", "tipologia"]

    def find_col(candidates):
        for c in candidates:
            if c in lowered:
                return lowered[c]
        return None

    name_col = find_col(possible_name_cols)
    url_col = find_col(possible_url_cols)
    desc_col = find_col(possible_desc_cols)
    cat_col = find_col(possible_cat_cols)

    if not name_col:
        raise ValueError(
            "Nel CSV serve almeno una colonna nome prodotto. "
            "Colonne accettate ad esempio: name, nome, title, product_name."
        )

    work = df.copy()
    work["__name"] = work[name_col].fillna("").astype(str)
    work["__norm_name"] = work["__name"].apply(normalize_text)

    if url_col:
        work["__url"] = work[url_col].fillna("").astype(str)
    else:
        work["__url"] = ""

    if desc_col:
        work["__desc"] = work[desc_col].fillna("").astype(str)
    else:
        work["__desc"] = ""

    if cat_col:
        work["__cat"] = work[cat_col].fillna("").astype(str)
    else:
        work["__cat"] = ""

    return work, {
        "name_col": name_col,
        "url_col": url_col,
        "desc_col": desc_col,
        "cat_col": cat_col,
        "all_columns": original_columns,
    }

def search_products(df: pd.DataFrame, user_text: str, top_k: int = 5):
    query = normalize_text(user_text)
    results = []

    for _, row in df.iterrows():
        name = row["__norm_name"]
        desc = normalize_text(row["__desc"])
        cat = normalize_text(row["__cat"])

        score_name = fuzz.token_set_ratio(query, name)
        score_partial = fuzz.partial_ratio(query, name)
        score_desc = fuzz.partial_ratio(query, desc) if desc else 0
        score_cat = fuzz.partial_ratio(query, cat) if cat else 0

        final_score = max(score_name, score_partial, score_desc, score_cat)

        results.append({
            "score": final_score,
            "name": row["__name"],
            "url": row["__url"],
            "description": row["__desc"],
            "category": row["__cat"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# =========================
# OPENAI CALLS
# =========================
PRODUCT_SYSTEM_PROMPT = f"""
Sei l'assistente di MGFishing.

Regole:
- Se l'utente chiede consigli o informazioni su prodotti, usa SOLO il contesto del CSV fornito.
- Non inventare prodotti che non sono nel contesto.
- Non dare link prodotto diretto all'utente.
- Se l'utente chiede il link o dove acquistarlo, devi dire di contattare {WHATSAPP_LABEL} su WhatsApp e fornire il link WhatsApp.
- Se hai trovato prodotti compatibili o simili, descrivili in modo utile e sintetico.
- Scrivi in italiano.
- Non citare altri siti.
"""

GENERAL_SYSTEM_PROMPT = f"""
Sei l'assistente di MGFishing.

Regole:
- Se l'utente chiede meteo, montature, tecniche, consigli di pesca o informazioni generali, puoi usare la ricerca web.
- Fornisci una risposta corretta, pratica e chiara.
- Non citare altri siti.
- Non scrivere fonti.
- Non nominare siti esterni.
- Se non sei sicuro o la richiesta richiede assistenza diretta, indirizza a {WHATSAPP_LABEL} su WhatsApp.
- Scrivi in italiano.
"""

def answer_from_products(user_text: str, product_matches: list[dict]) -> str:
    if not client:
        top = product_matches[:3]
        if not top or top[0]["score"] < 55:
            return fallback_whatsapp_reply(user_text)

        text = "Ho trovato questi prodotti nel catalogo MGFishing:\n\n"
        for p in top:
            text += f"- **{p['name']}**"
            if p["category"]:
                text += f" — {p['category']}"
            if p["description"]:
                short_desc = p["description"][:220].strip()
                text += f"\n  {short_desc}"
            text += "\n"
        text += f"\nPer il link o per maggiori informazioni ti consiglio di contattare **{WHATSAPP_LABEL}**:\n[Scrivi su WhatsApp]({make_whatsapp_message(user_text)})"
        return text

    catalog_context = "\n\n".join([
        f"Prodotto: {p['name']}\nCategoria: {p['category']}\nDescrizione: {p['description']}\nURL interno: {p['url']}"
        for p in product_matches[:5]
    ])

    user_prompt = f"""
Richiesta cliente:
{user_text}

Prodotti trovati nel CSV:
{catalog_context}

Istruzioni:
- Rispondi solo usando questi prodotti.
- Se l'utente chiede consigli, suggerisci il prodotto o i prodotti più adatti.
- Se l'utente chiede il link, NON dare il link prodotto: invitalo a contattare {WHATSAPP_LABEL} su WhatsApp usando questo link:
{make_whatsapp_message(user_text)}
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.output_text.strip()
    except Exception:
        return fallback_whatsapp_reply(user_text)

def answer_general_with_web(user_text: str) -> str:
    if not client:
        return fallback_whatsapp_reply(user_text)

    user_prompt = f"""
Richiesta cliente:
{user_text}

Rispondi in modo utile e pratico.
Puoi usare la ricerca web.
Non citare siti esterni.
Non inserire fonti.
Se non sei sicuro, invita a contattare {WHATSAPP_LABEL} su WhatsApp:
{make_whatsapp_message(user_text)}
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[{"type": "web_search_preview"}],
            temperature=0.4,
        )
        text = response.output_text.strip()
        if text:
            return text
    except Exception:
        pass

    return fallback_whatsapp_reply(user_text)

# =========================
# MAIN LOGIC
# =========================
def generate_answer(user_text: str, catalog_df: pd.DataFrame | None) -> str:
    clean = normalize_text(user_text)

    if clean in ["ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"]:
        return (
            "Ciao! Sono l’assistente MGFishing 🎣\n\n"
            "Posso aiutarti con:\n"
            "- consigli sui prodotti\n"
            "- informazioni generali di pesca\n"
            "- meteo e montature\n\n"
            f"Per link prodotto o assistenza diretta puoi contattare **{WHATSAPP_LABEL}** qui:\n"
            f"[Scrivi su WhatsApp]({make_whatsapp_message()})"
        )

    if is_link_request(user_text):
        return (
            f"Per il link del prodotto ti consiglio di contattare direttamente **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
            f"[Scrivi su WhatsApp]({make_whatsapp_message(user_text)})"
        )

    if is_product_request(user_text) and catalog_df is not None:
        matches = search_products(catalog_df, user_text, top_k=5)
        if matches and matches[0]["score"] >= 45:
            return answer_from_products(user_text, matches)
        return fallback_whatsapp_reply(user_text)

    if is_general_info_request(user_text):
        return answer_general_with_web(user_text)

    # fallback intelligente
    if catalog_df is not None:
        matches = search_products(catalog_df, user_text, top_k=5)
        if matches and matches[0]["score"] >= 60:
            return answer_from_products(user_text, matches)

    return answer_general_with_web(user_text)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Catalogo prodotti CSV")
    uploaded_file = st.file_uploader("Carica il CSV prodotti", type=["csv"])

    catalog_df = None
    catalog_info = None

    if uploaded_file is not None:
        try:
            catalog_df, catalog_info = load_catalog_from_csv(uploaded_file)
            st.success("CSV caricato correttamente")
            st.write(f"Colonna nome: **{catalog_info['name_col']}**")
            if catalog_info["url_col"]:
                st.write(f"Colonna link: **{catalog_info['url_col']}**")
            if catalog_info["desc_col"]:
                st.write(f"Colonna descrizione: **{catalog_info['desc_col']}**")
            if catalog_info["cat_col"]:
                st.write(f"Colonna categoria: **{catalog_info['cat_col']}**")
            st.write(f"Prodotti letti: **{len(catalog_df)}**")
        except Exception as e:
            st.error(f"Errore CSV: {e}")
            catalog_df = None
    else:
        st.info("Carica il CSV per attivare i consigli sui prodotti.")

    if not OPENAI_API_KEY:
        st.warning("Manca OPENAI_API_KEY nei secrets o nelle variabili ambiente.")

# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono il chatbot MGFishing 🎣\n\n"
                "Posso aiutarti con prodotti, consigli di pesca, meteo e montature.\n"
                f"Per assistenza diretta puoi scrivere a **{WHATSAPP_LABEL}**:\n"
                f"[Scrivi su WhatsApp]({make_whatsapp_message()})"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Scrivi la tua domanda...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta migliore..."):
            reply = generate_answer(prompt, catalog_df)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
