import os
import re
import urllib.parse
import pandas as pd
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz

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

def whatsapp_link(user_text: str = "") -> str:
    msg = "Ciao Emanuele, vorrei informazioni su un prodotto MGFishing."
    if user_text:
        msg = f"Ciao Emanuele, vorrei informazioni su: {user_text}"
    return f"{WHATSAPP_URL}?text={urllib.parse.quote(msg)}"

def whatsapp_reply(user_text: str = "") -> str:
    return (
        f"Per maggiori informazioni ti consiglio di contattare direttamente **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
        f"[Scrivi su WhatsApp]({whatsapp_link(user_text)})"
    )

def is_link_request(text: str) -> bool:
    t = normalize_text(text)
    keys = [
        "link", "mandami il link", "mi mandi il link", "dammi il link",
        "url", "pagina prodotto", "dove lo trovo", "acquistarlo", "comprarlo"
    ]
    return any(k in t for k in keys)

def is_general_info_request(text: str) -> bool:
    t = normalize_text(text)
    keys = [
        "meteo", "vento", "mare", "pioggia", "pressione", "temperature",
        "montatura", "montature", "terminale", "trave", "bracciolo",
        "come pescare", "tecnica", "tecniche", "consiglio pesca",
        "surfcasting", "spinning", "bolognese", "feeder", "bolentino",
        "trota lago", "carpfishing", "serra", "spigola", "orata"
    ]
    return any(k in t for k in keys)

def likely_product_question(text: str) -> bool:
    t = normalize_text(text)
    keys = [
        "mulinello", "mulinelli", "canna", "canne", "trecciato", "monofilo",
        "fluorocarbon", "esca", "esche", "artificiale", "artificiali",
        "bombarda", "galleggiante", "amo", "ami", "girella", "girelle",
        "shimano", "daiwa", "trabucco", "colmic", "tubertini", "molix",
        "rapture", "major craft", "yuki", "prodotto", "prodotti", "articolo",
        "misura", "taglia", "grammi", "azione", "frizione", "bobina"
    ]
    return any(k in t for k in keys)

# =========================
# CSV
# =========================
@st.cache_data(show_spinner=False)
def load_catalog_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    cols = list(df.columns)
    lowered = {c.lower().strip(): c for c in cols}

    def find_col(candidates):
        for c in candidates:
            if c in lowered:
                return lowered[c]
        return None

    name_col = find_col(["name", "nome", "product_name", "title", "titolo", "prodotto"])
    desc_col = find_col(["description", "descrizione", "short_description", "descrizione_breve"])
    cat_col = find_col(["category", "categoria", "brand", "marchio", "tipologia"])
    sku_col = find_col(["sku", "reference", "riferimento", "codice"])
    price_col = find_col(["price", "prezzo", "final_price"])

    if not name_col:
        raise ValueError("Nel CSV manca una colonna nome prodotto, ad esempio: name, nome, title, product_name")

    work = df.copy()
    work["__name"] = work[name_col].fillna("").astype(str)
    work["__norm_name"] = work["__name"].apply(normalize_text)
    work["__desc"] = work[desc_col].fillna("").astype(str) if desc_col else ""
    work["__norm_desc"] = work["__desc"].apply(normalize_text)
    work["__cat"] = work[cat_col].fillna("").astype(str) if cat_col else ""
    work["__norm_cat"] = work["__cat"].apply(normalize_text)
    work["__sku"] = work[sku_col].fillna("").astype(str) if sku_col else ""
    work["__price"] = work[price_col].fillna("").astype(str) if price_col else ""

    return work, {
        "name_col": name_col,
        "desc_col": desc_col,
        "cat_col": cat_col,
        "sku_col": sku_col,
        "price_col": price_col,
        "all_columns": cols,
    }

def extract_keywords(text: str):
    t = normalize_text(text)
    stopwords = {
        "mi","puoi","consigliare","consigli","consiglio","una","uno","un","dei","delle","della","del",
        "per","da","dai","delle","di","che","con","su","il","lo","la","gli","le","e","o","ho","vorrei",
        "cerco","cerco","serve","servono","mi","puo","potresti","adatta","adatto","adatte","adatti"
    }
    words = [w for w in t.split() if len(w) >= 3 and w not in stopwords]
    return words

def search_products_strict(df: pd.DataFrame, user_text: str, top_k: int = 5):
    query = normalize_text(user_text)
    keywords = extract_keywords(user_text)
    results = []

    for _, row in df.iterrows():
        name = row["__norm_name"]
        desc = row["__norm_desc"]
        cat = row["__norm_cat"]
        sku = normalize_text(row["__sku"])

        score_name = fuzz.token_set_ratio(query, name)
        score_partial = fuzz.partial_ratio(query, name)
        score_desc = fuzz.partial_ratio(query, desc) if desc else 0
        score_cat = fuzz.partial_ratio(query, cat) if cat else 0
        score_sku = fuzz.partial_ratio(query, sku) if sku else 0

        keyword_hits = 0
        for kw in keywords:
            if kw in name or kw in desc or kw in cat or kw in sku:
                keyword_hits += 1

        # punteggio finale più severo
        final_score = max(score_name, score_partial, score_desc, score_cat, score_sku)
        if keyword_hits:
            final_score += min(keyword_hits * 4, 12)

        results.append({
            "score": final_score,
            "keyword_hits": keyword_hits,
            "name": row["__name"],
            "description": row["__desc"],
            "category": row["__cat"],
            "sku": row["__sku"],
            "price": row["__price"],
        })

    results.sort(key=lambda x: (x["score"], x["keyword_hits"]), reverse=True)
    return results[:top_k]

def format_product_reply(matches, user_text):
    strong = []
    for m in matches:
        # filtri severi per evitare invenzioni
        if m["score"] >= 78 and m["keyword_hits"] >= 1:
            strong.append(m)

    if not strong:
        return (
            "Al momento non riesco a identificare con certezza un prodotto preciso nel catalogo MGFishing in base alla tua richiesta.\n\n"
            f"Per evitare di indicarti articoli sbagliati, ti consiglio di contattare direttamente **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
            f"[Scrivi su WhatsApp]({whatsapp_link(user_text)})"
        )

    text = "Ti consiglio questi prodotti presenti nel catalogo MGFishing:\n\n"
    for p in strong[:3]:
        text += f"**{p['name']}**\n"
        if p["category"]:
            text += f"- Categoria: {p['category']}\n"
        if p["price"]:
            text += f"- Prezzo: {p['price']}\n"
        if p["description"]:
            short_desc = p["description"][:220].strip()
            text += f"- Dettagli: {short_desc}\n"
        text += "\n"

    text += (
        f"Per disponibilità, conferma modello preciso o assistenza diretta ti consiglio di contattare **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
        f"[Scrivi su WhatsApp]({whatsapp_link(user_text)})"
    )
    return text

# =========================
# OPENAI SOLO PER INFO GENERALI
# =========================
GENERAL_SYSTEM_PROMPT = f"""
Sei l'assistente di MGFishing.

Regole:
- Usa il web solo per domande generali come meteo, montature, tecniche, consigli di pesca.
- Non citare fonti.
- Non nominare siti esterni.
- Scrivi in italiano in modo chiaro e pratico.
- Se la richiesta è troppo specifica o non sei sicuro, invita a contattare {WHATSAPP_LABEL} su WhatsApp.
"""

def answer_general_with_web(user_text: str) -> str:
    if not client:
        return whatsapp_reply(user_text)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tools=[{"type": "web_search_preview"}],
            temperature=0.3,
        )
        text = response.output_text.strip()
        if text:
            return text
    except Exception:
        pass

    return whatsapp_reply(user_text)

# =========================
# LOGICA PRINCIPALE
# =========================
def generate_answer(user_text: str, catalog_df: pd.DataFrame | None) -> str:
    clean = normalize_text(user_text)

    if clean in ["ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"]:
        return (
            "Ciao! Sono l’assistente MGFishing 🎣\n\n"
            "Posso aiutarti con:\n"
            "- consigli sui prodotti presenti nel catalogo\n"
            "- meteo e montature\n"
            "- consigli generali di pesca\n\n"
            f"Per assistenza diretta puoi scrivere a **{WHATSAPP_LABEL}**:\n"
            f"[Scrivi su WhatsApp]({whatsapp_link()})"
        )

    if is_link_request(user_text):
        return (
            f"Per il link o per acquistare il prodotto ti consiglio di contattare direttamente **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
            f"[Scrivi su WhatsApp]({whatsapp_link(user_text)})"
        )

    # Se sembra una richiesta prodotto, usa SOLO il CSV
    if likely_product_question(user_text):
        if catalog_df is None:
            return (
                "Per consigliarti correttamente i prodotti serve il catalogo CSV caricato nell'app.\n\n"
                f"Nel frattempo puoi contattare **{WHATSAPP_LABEL}** su WhatsApp:\n\n"
                f"[Scrivi su WhatsApp]({whatsapp_link(user_text)})"
            )

        matches = search_products_strict(catalog_df, user_text, top_k=5)
        return format_product_reply(matches, user_text)

    # Domande generali
    if is_general_info_request(user_text):
        return answer_general_with_web(user_text)

    # fallback: prima prova prodotto in modo severo, altrimenti info generali
    if catalog_df is not None:
        matches = search_products_strict(catalog_df, user_text, top_k=5)
        if matches and matches[0]["score"] >= 82 and matches[0]["keyword_hits"] >= 1:
            return format_product_reply(matches, user_text)

    return answer_general_with_web(user_text)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Catalogo prodotti CSV")
    uploaded_file = st.file_uploader("Carica il CSV prodotti", type=["csv"])

    catalog_df = None
    if uploaded_file is not None:
        try:
            catalog_df, info = load_catalog_from_csv(uploaded_file)
            st.success("CSV caricato correttamente")
            st.write(f"Prodotti letti: **{len(catalog_df)}**")
            st.write(f"Colonna nome: **{info['name_col']}**")
            if info["desc_col"]:
                st.write(f"Colonna descrizione: **{info['desc_col']}**")
            if info["cat_col"]:
                st.write(f"Colonna categoria: **{info['cat_col']}**")
            if info["sku_col"]:
                st.write(f"Colonna SKU/codice: **{info['sku_col']}**")
            if info["price_col"]:
                st.write(f"Colonna prezzo: **{info['price_col']}**")
        except Exception as e:
            st.error(f"Errore CSV: {e}")
            catalog_df = None
    else:
        st.info("Carica il CSV per attivare i consigli sui prodotti.")

    if not OPENAI_API_KEY:
        st.warning("Manca OPENAI_API_KEY.")

# =========================
# CHAT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono il chatbot MGFishing 🎣\n\n"
                "Posso aiutarti con prodotti, meteo, montature e consigli di pesca.\n\n"
                f"Per assistenza diretta puoi scrivere a **{WHATSAPP_LABEL}**:\n"
                f"[Scrivi su WhatsApp]({whatsapp_link()})"
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
