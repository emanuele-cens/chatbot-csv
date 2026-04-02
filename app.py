import os
import re
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MGFishing Chatbot", page_icon="🎣", layout="centered")

WHATSAPP_NUMBER = "393494166335"
WHATSAPP_LINK = f"https://wa.me/{WHATSAPP_NUMBER}"
CSV_PATH = "prodotti.csv"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Modello per risposte generali
GENERAL_MODEL = "gpt-5-mini"

# =========================
# STILI
# =========================
st.markdown("""
<style>
.block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 2rem;}
.small-muted {color: #6b7280; font-size: 0.92rem;}
.whatsapp-box {
    padding: 12px 14px;
    border-radius: 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    margin-top: 10px;
}
.product-card {
    padding: 14px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UTILS
# =========================
def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9àèéìòùçñü\s\-\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    text = normalize_text(text)
    return [t for t in re.split(r"[\s\-/]+", text) if len(t) > 1]

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def first_existing_column(df, candidates):
    cols_norm = {normalize_text(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]

    for c in df.columns:
        c_norm = normalize_text(c)
        for cand in candidates:
            cand_norm = normalize_text(cand)
            if cand_norm in c_norm or c_norm in cand_norm:
                return c
    return None

def build_whatsapp_message(text="Ciao Emanuele, vorrei assistenza su un prodotto."):
    from urllib.parse import quote
    return f"{WHATSAPP_LINK}?text={quote(text)}"

# =========================
# CARICAMENTO CSV
# =========================
@st.cache_data(show_spinner=False)
def load_catalog():
    if not os.path.exists(CSV_PATH):
        return None, "File prodotti.csv non trovato nella root del progetto."

    attempts = [
        {"sep": ","},
        {"sep": ";"},
        {"sep": None, "engine": "python"},
    ]

    last_error = None
    for params in attempts:
        try:
            df = pd.read_csv(CSV_PATH, **params)
            if df is not None and len(df.columns) > 0 and len(df) > 0:
                df.columns = [str(c).strip() for c in df.columns]
                return df, None
        except Exception as e:
            last_error = str(e)

    return None, f"Impossibile leggere prodotti.csv. Errore: {last_error}"

catalog_df, catalog_error = load_catalog()

# =========================
# PREPARAZIONE CATALOGO
# =========================
def prepare_catalog(df):
    if df is None or df.empty:
        return None

    name_col = first_existing_column(df, [
        "nome", "name", "titolo", "title", "prodotto", "product", "nome prodotto"
    ])
    desc_col = first_existing_column(df, [
        "descrizione", "description", "descrizione breve", "short description", "testo"
    ])
    brand_col = first_existing_column(df, [
        "marca", "brand", "produttore", "manufacturer"
    ])
    category_col = first_existing_column(df, [
        "categoria", "category", "famiglia", "reparto"
    ])
    price_col = first_existing_column(df, [
        "prezzo", "price", "prezzo ivato", "final price", "sale price"
    ])
    sku_col = first_existing_column(df, [
        "sku", "reference", "codice", "ean", "id"
    ])
    url_col = first_existing_column(df, [
        "url", "link", "product url", "permalink", "href"
    ])
    stock_col = first_existing_column(df, [
        "stock", "disponibilita", "availability", "qty", "quantita"
    ])

    prepared = df.copy()

    prepared["_name"] = prepared[name_col].fillna("").astype(str) if name_col else ""
    prepared["_desc"] = prepared[desc_col].fillna("").astype(str) if desc_col else ""
    prepared["_brand"] = prepared[brand_col].fillna("").astype(str) if brand_col else ""
    prepared["_category"] = prepared[category_col].fillna("").astype(str) if category_col else ""
    prepared["_price"] = prepared[price_col].fillna("").astype(str) if price_col else ""
    prepared["_sku"] = prepared[sku_col].fillna("").astype(str) if sku_col else ""
    prepared["_url"] = prepared[url_col].fillna("").astype(str) if url_col else ""
    prepared["_stock"] = prepared[stock_col].fillna("").astype(str) if stock_col else ""

    prepared["_search_blob"] = (
        prepared["_name"].fillna("") + " " +
        prepared["_brand"].fillna("") + " " +
        prepared["_category"].fillna("") + " " +
        prepared["_desc"].fillna("") + " " +
        prepared["_sku"].fillna("")
    ).astype(str)

    prepared["_search_blob_norm"] = prepared["_search_blob"].apply(normalize_text)

    return prepared

catalog_prepared = prepare_catalog(catalog_df)

# =========================
# RICONOSCIMENTO INTENTO
# =========================
PRODUCT_KEYWORDS = [
    "prodotto", "prodotti", "canna", "canne", "mulinello", "mulinelli", "trecciato",
    "nylon", "fluorocarbon", "esca", "esche", "artificiale", "artificiali",
    "pastura", "bombarda", "galleggiante", "amo", "ami", "girella", "girelle",
    "finale", "terminale", "combo", "kit", "fishing", "trabucco", "daiwa",
    "shimano", "colmic", "tubertini", "molix", "rapture", "major craft",
    "trota", "surfcasting", "bolognese", "feeder", "spinning", "eging", "carpfishing"
]

LINK_KEYWORDS = [
    "link", "mandami il link", "invia il link", "dammi il link",
    "url", "pagina prodotto", "apri prodotto", "scheda prodotto"
]

GENERAL_WEB_KEYWORDS = [
    "meteo", "tempo", "vento", "mare", "onde", "pressione", "pioggia",
    "montatura", "montature", "trave", "terminale", "finale", "innesco",
    "consiglio", "consigli", "come pescare", "tecnica", "tecniche",
    "nodo", "nodi", "lancio", "assetto", "mareggiata", "luna", "stagione"
]

OPERATOR_KEYWORDS = [
    "operatore", "umano", "persona", "assistenza", "whatsapp", "emanuele", "contatto"
]

SHIPPING_KEYWORDS = [
    "spedizione", "tracking", "tracciamento", "ordine", "consegna", "pacco", "corriere"
]

def contains_any(text, keywords):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in keywords)

def detect_intent(user_text):
    t = normalize_text(user_text)

    if contains_any(t, OPERATOR_KEYWORDS):
        return "operator"

    if contains_any(t, LINK_KEYWORDS):
        return "link"

    if contains_any(t, SHIPPING_KEYWORDS):
        return "shipping"

    if contains_any(t, GENERAL_WEB_KEYWORDS):
        return "general_web"

    if contains_any(t, PRODUCT_KEYWORDS):
        return "product"

    return "general_web"

# =========================
# MATCH PRODOTTI DA CSV
# =========================
def score_product_match(query, row):
    q_norm = normalize_text(query)
    q_tokens = set(tokenize(query))

    blob = row["_search_blob_norm"]
    name_norm = normalize_text(row["_name"])
    brand_norm = normalize_text(row["_brand"])
    category_norm = normalize_text(row["_category"])

    score = 0

    if q_norm and q_norm in name_norm:
        score += 120
    if q_norm and q_norm in blob:
        score += 80

    for tok in q_tokens:
        if tok in name_norm:
            score += 18
        elif tok in brand_norm:
            score += 12
        elif tok in category_norm:
            score += 10
        elif tok in blob:
            score += 5

    if row["_name"] and normalize_text(row["_name"]).startswith(q_norm[:20]):
        score += 20

    # Premia brand noti se presenti nella domanda
    for brand in ["trabucco", "daiwa", "shimano", "colmic", "tubertini", "rapture", "molix", "major craft"]:
        if brand in q_norm and brand in blob:
            score += 25

    return score

def search_products(query, df, top_n=5):
    if df is None or df.empty:
        return []

    scored = []
    for idx, row in df.iterrows():
        score = score_product_match(query, row)
        if score > 0:
            scored.append((score, idx, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Tieni solo risultati sensati
    results = []
    seen = set()
    for score, idx, row in scored:
        name = safe_str(row["_name"])
        if not name:
            continue
        key = normalize_text(name)
        if key in seen:
            continue
        seen.add(key)
        results.append((score, row))
        if len(results) >= top_n:
            break

    return results

def format_product_card(row):
    name = safe_str(row["_name"])
    brand = safe_str(row["_brand"])
    category = safe_str(row["_category"])
    price = safe_str(row["_price"])
    desc = safe_str(row["_desc"])
    sku = safe_str(row["_sku"])
    stock = safe_str(row["_stock"])

    details = []
    if brand:
        details.append(f"Marca: {brand}")
    if category:
        details.append(f"Categoria: {category}")
    if price:
        details.append(f"Prezzo: {price}")
    if sku:
        details.append(f"Codice: {sku}")
    if stock:
        details.append(f"Disponibilità: {stock}")

    extra = " • ".join(details)

    short_desc = desc[:280].strip()
    if len(desc) > 280:
        short_desc += "..."

    html = f"""
    <div class="product-card">
        <div><strong>{name}</strong></div>
        {"<div class='small-muted'>" + extra + "</div>" if extra else ""}
        {"<div style='margin-top:8px;'>" + short_desc + "</div>" if short_desc else ""}
    </div>
    """
    return html

# =========================
# RISPOSTE FISSE
# =========================
def whatsapp_response(custom_text=None):
    msg = custom_text or "Ciao Emanuele, vorrei assistenza."
    link = build_whatsapp_message(msg)
    return f"""Per assistenza diretta puoi contattare **Emanuele** su WhatsApp:

[{link}]({link})"""

def shipping_response():
    return f"""Per informazioni su **spedizione, tracking, consegna o stato ordine** ti consiglio di contattare direttamente **Emanuele** su WhatsApp:

[{build_whatsapp_message("Ciao Emanuele, vorrei informazioni su spedizione o tracking del mio ordine.")}]({build_whatsapp_message("Ciao Emanuele, vorrei informazioni su spedizione o tracking del mio ordine.")})"""

def link_response(user_text):
    link = build_whatsapp_message(f"Ciao Emanuele, mi serve il link del prodotto: {user_text}")
    return f"""Per ricevere il **link corretto del prodotto** scrivi direttamente a **Emanuele** su WhatsApp:

[{link}]({link})"""

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

GENERAL_SYSTEM_PROMPT = """
Sei l'assistente di MGFishing.

Regole obbligatorie:
- Rispondi sempre in italiano.
- Quando rispondi a domande generali su pesca, meteo, montature, tecniche e consigli pratici, dai una risposta chiara, utile e concreta.
- Non citare nomi di siti esterni.
- Non scrivere frasi tipo "secondo il sito..." o "ho trovato su...".
- Non inventare prodotti del catalogo MGFishing.
- Se la richiesta riguarda un prodotto specifico del catalogo, non devi rispondere tu: quella parte viene gestita separatamente.
- Se la domanda è ambigua o molto specifica e non sei sicuro, invita a contattare Emanuele su WhatsApp al 3494166335.
- Mantieni tono professionale, utile e commerciale.
"""

def ask_openai_general(question):
    if not client:
        return f"""Al momento il servizio AI non è disponibile.

Puoi contattare direttamente **Emanuele** su WhatsApp:
[{build_whatsapp_message("Ciao Emanuele, vorrei assistenza.")}]({build_whatsapp_message("Ciao Emanuele, vorrei assistenza.")})"""

    try:
        response = client.responses.create(
            model=GENERAL_MODEL,
            input=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        )
        text = getattr(response, "output_text", None)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return f"""Non sono riuscito a rispondere con precisione.

Puoi contattare direttamente **Emanuele** su WhatsApp:
[{build_whatsapp_message("Ciao Emanuele, vorrei assistenza su questa richiesta.")}]({build_whatsapp_message("Ciao Emanuele, vorrei assistenza su questa richiesta.")})"""

# =========================
# RISPOSTE SU PRODOTTI DA CSV
# =========================
def answer_product_question(user_text):
    if catalog_prepared is None or catalog_prepared.empty:
        return {
            "type": "text",
            "content": f"""Il catalogo prodotti al momento non è disponibile nell'app.

Per assistenza diretta puoi scrivere a **Emanuele** su WhatsApp:
[{build_whatsapp_message("Ciao Emanuele, mi serve aiuto per trovare un prodotto.")}]({build_whatsapp_message("Ciao Emanuele, mi serve aiuto per trovare un prodotto.")})"""
        }

    matches = search_products(user_text, catalog_prepared, top_n=5)

    if not matches:
        return {
            "type": "text",
            "content": f"""Non ho trovato nel catalogo un prodotto chiaramente collegato a questa richiesta.

Per una verifica rapida ti consiglio di scrivere a **Emanuele** su WhatsApp:
[{build_whatsapp_message(f"Ciao Emanuele, sto cercando questo prodotto: {user_text}")}]({build_whatsapp_message(f"Ciao Emanuele, sto cercando questo prodotto: {user_text}")})"""
        }

    # Se la domanda sembra proprio una richiesta di consiglio
    advice_prefix = "In base al catalogo MGFishing, questi prodotti sembrano i più adatti alla tua richiesta:"
    if any(k in normalize_text(user_text) for k in ["consiglia", "consigliami", "quale", "migliore", "adatto", "scegliere"]):
        advice_prefix = "In base ai prodotti presenti nel catalogo MGFishing, ti consiglierei di valutare questi articoli:"

    products = [row for score, row in matches]
    return {
        "type": "products",
        "intro": advice_prefix,
        "products": products,
        "footer": f"""Se vuoi il **link esatto del prodotto** o conferma immediata della disponibilità, contatta **Emanuele** su WhatsApp:
[{build_whatsapp_message(f"Ciao Emanuele, mi serve il link o conferma disponibilità per questo prodotto: {user_text}")}]({build_whatsapp_message(f"Ciao Emanuele, mi serve il link o conferma disponibilità per questo prodotto: {user_text}")})"""
    }

# =========================
# MOTORE PRINCIPALE
# =========================
def generate_answer(user_text):
    intent = detect_intent(user_text)

    if intent == "operator":
        return {"type": "text", "content": whatsapp_response("Ciao Emanuele, vorrei parlare con un operatore.")}

    if intent == "link":
        return {"type": "text", "content": link_response(user_text)}

    if intent == "shipping":
        return {"type": "text", "content": shipping_response()}

    if intent == "product":
        return answer_product_question(user_text)

    if intent == "general_web":
        text = ask_openai_general(user_text)

        # Se la risposta torna vuota o troppo debole, fallback WhatsApp
        if not text or len(text.strip()) < 8:
            text = whatsapp_response("Ciao Emanuele, mi serve assistenza.")
        return {"type": "text", "content": text}

    return {"type": "text", "content": whatsapp_response("Ciao Emanuele, mi serve assistenza.")}

# =========================
# UI
# =========================
st.title("🎣 MGFishing Chatbot")

st.markdown("""
Ciao! Sono il chatbot MGFishing 🎣

Posso aiutarti con:
- consigli sui prodotti presenti nel catalogo
- meteo, montature e tecniche di pesca
- assistenza diretta tramite WhatsApp
""")

if catalog_prepared is not None:
    st.caption(f"Catalogo collegato: {len(catalog_prepared)} prodotti")
else:
    st.caption("Catalogo non disponibile")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": f"""Ciao! Sono l’assistente MGFishing 🎣

Posso aiutarti con prodotti, meteo, montature e consigli di pesca.

Per assistenza diretta puoi scrivere a **Emanuele** su WhatsApp:
[{build_whatsapp_message("Ciao Emanuele, vorrei assistenza.")}]({build_whatsapp_message("Ciao Emanuele, vorrei assistenza.")})"""
        }
    ]

# mostra chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "products":
            st.markdown(msg["intro"])
            for row in msg["products"]:
                st.markdown(format_product_card(row), unsafe_allow_html=True)
            st.markdown(msg["footer"])
        else:
            st.markdown(msg["content"])

# input
user_prompt = st.chat_input("Scrivi la tua domanda...")

if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto preparando la risposta..."):
            answer = generate_answer(user_prompt)

            if answer["type"] == "products":
                st.markdown(answer["intro"])
                for row in answer["products"]:
                    st.markdown(format_product_card(row), unsafe_allow_html=True)
                st.markdown(answer["footer"])
            else:
                st.markdown(answer["content"])

    st.session_state.messages.append({
        "role": "assistant",
        **answer
    })
