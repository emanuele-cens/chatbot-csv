import os
import re
import io
import csv
import math
import json
import time
import html
import unicodedata
from typing import List, Dict, Tuple

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="MGFishing Chatbot", page_icon="🎣", layout="centered")

CSV_PATH = "prodotti.csv"
KNOWLEDGE_PATH = "knowledge.txt"
WHATSAPP_NUMBER = "393494166335"
WHATSAPP_LABEL = "Emanuele"

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# =========================================================
# UTILS
# =========================================================
def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9\s€.,/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [t for t in re.findall(r"[a-z0-9€]+", text) if len(t) > 1]

def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def whatsapp_link(message: str) -> str:
    from urllib.parse import quote
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(message)}"

def safe_float(value):
    if value is None:
        return None
    s = str(value).strip().replace("€", "").replace("EUR", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# =========================================================
# CSV LOADING ROBUSTO
# =========================================================
def try_read_csv_with_sep(content: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(content),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    )

@st.cache_data(show_spinner=False)
def load_catalog(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()

    candidates = [",", ";", "\t", "|"]
    best_df = None
    best_cols = 0

    for sep in candidates:
        try:
            df = try_read_csv_with_sep(content, sep)
            if df is not None and len(df.columns) > best_cols:
                best_df = df
                best_cols = len(df.columns)
        except Exception:
            pass

    if best_df is None or best_df.empty:
        try:
            best_df = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            return pd.DataFrame()

    # pulizia colonne
    best_df.columns = [str(c).strip() for c in best_df.columns]
    best_df = best_df.fillna("")

    # nomi colonna possibili
    colmap = {}
    lower_cols = {c.lower(): c for c in best_df.columns}

    def find_col(possibles):
        for p in possibles:
            for c in best_df.columns:
                if c.lower().strip() == p.lower().strip():
                    return c
        for p in possibles:
            for c in best_df.columns:
                if p.lower().strip() in c.lower().strip():
                    return c
        return None

    colmap["name"] = find_col(["name", "nome", "titolo", "title", "product_name", "nome prodotto"])
    colmap["description"] = find_col(["description", "descrizione", "short_description", "desc"])
    colmap["price"] = find_col(["price", "prezzo", "prezzo iva incl", "prezzo finale", "final_price"])
    colmap["url"] = find_col(["url", "link", "product_url", "permalink", "href"])
    colmap["category"] = find_col(["category", "categoria", "cat"])
    colmap["brand"] = find_col(["brand", "marca", "manufacturer"])

    # fallback furbo: se manca name prova prima colonna utile
    if not colmap["name"] and len(best_df.columns) > 0:
        colmap["name"] = best_df.columns[0]

    # crea dataframe standard
    std = pd.DataFrame()
    std["name"] = best_df[colmap["name"]] if colmap["name"] in best_df.columns else ""
    std["description"] = best_df[colmap["description"]] if colmap["description"] in best_df.columns else ""
    std["price"] = best_df[colmap["price"]] if colmap["price"] in best_df.columns else ""
    std["url"] = best_df[colmap["url"]] if colmap["url"] in best_df.columns else ""
    std["category"] = best_df[colmap["category"]] if colmap["category"] in best_df.columns else ""
    std["brand"] = best_df[colmap["brand"]] if colmap["brand"] in best_df.columns else ""

    std = std.fillna("")
    std = std[std["name"].astype(str).str.strip() != ""].copy()

    std["name_norm"] = std["name"].apply(normalize_text)
    std["desc_norm"] = std["description"].apply(normalize_text)
    std["category_norm"] = std["category"].apply(normalize_text)
    std["brand_norm"] = std["brand"].apply(normalize_text)
    std["all_text"] = (
        std["name_norm"] + " " + std["desc_norm"] + " " + std["category_norm"] + " " + std["brand_norm"]
    ).str.strip()

    std["price_num"] = std["price"].apply(safe_float)

    # rimuove righe chiaramente sporche
    std = std[std["name_norm"].str.len() > 2].copy()

    return std.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_knowledge(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

catalog_df = load_catalog(CSV_PATH)
knowledge_text = load_knowledge(KNOWLEDGE_PATH)

# =========================================================
# INTENT
# =========================================================
PRODUCT_WORDS = {
    "canna", "canna da pesca", "canne", "mulinello", "mulinelli", "trecciato", "filo",
    "monofilo", "fluorocarbon", "amo", "ami", "artificiale", "artificiali", "esca", "esche",
    "bombarda", "galleggiante", "guadino", "testa piombata", "girella", "girelle", "piombo",
    "jig", "popper", "minnow", "feeder", "surfcasting", "bolognese", "spinning", "trota",
    "carpfishing", "ledgering", "bolentino", "eging", "seppia", "calamaro", "traina", "mare", "lago"
}

LINK_WORDS = {
    "link", "mandami il link", "inviami il link", "pagina prodotto", "scheda prodotto",
    "url", "apri prodotto", "dove lo trovo", "prodotto sul sito"
}

SHIPPING_WORDS = {
    "spedizione", "tracking", "tracciamento", "consegna", "tempi di spedizione",
    "quando arriva", "ordine", "stato ordine", "pagamento", "resi", "reso", "corriere"
}

WEB_ADVICE_WORDS = {
    "meteo", "vento", "mare", "onde", "marea", "pressione", "pioggia", "temperatura",
    "montatura", "montature", "trave", "finale", "terminali", "innesco", "esca", "tecnica",
    "come pescare", "consiglio pesca", "quando pescare", "spot", "orario", "luna"
}

GREETING_WORDS = {"ciao", "salve", "buongiorno", "buonasera", "hey"}

def detect_intent(query: str) -> str:
    q = normalize_text(query)

    if q in GREETING_WORDS or len(q) <= 6 and any(g in q for g in GREETING_WORDS):
        return "greeting"

    if any(k in q for k in LINK_WORDS):
        return "product_link"

    if any(k in q for k in SHIPPING_WORDS):
        return "shipping"

    if any(k in q for k in WEB_ADVICE_WORDS):
        # se contiene parole prodotto forti ma anche montature/meteo, prevale web advice
        return "web_advice"

    if any(k in q for k in PRODUCT_WORDS):
        return "product_advice"

    return "generic"

# =========================================================
# FILTRI E MATCH PRODOTTI
# =========================================================
CATEGORY_SYNONYMS = {
    "surfcasting": ["surfcasting", "surf", "beach ledgering"],
    "bolognese": ["bolognese", "bolo"],
    "spinning": ["spinning", "spin"],
    "trota": ["trota", "trout", "area trout", "tremarella"],
    "feeder": ["feeder"],
    "eging": ["eging", "calamaro", "seppia"],
    "bolentino": ["bolentino"],
    "carpfishing": ["carpfishing", "carp fishing", "carpa"],
    "ledgering": ["ledgering"],
    "mare": ["mare", "saltwater", "marino"],
    "lago": ["lago", "lake"],
    "canna": ["canna", "canne", "rod", "rods"],
    "mulinello": ["mulinello", "mulinelli", "reel", "reels"],
    "filo": ["filo", "monofilo", "trecciato", "fluorocarbon"],
    "artificiale": ["artificiale", "artificiali", "esca", "esche", "minnow", "popper", "jig"],
}

STOPWORDS = {
    "mi", "puoi", "puoi", "consiglia", "consigliami", "consiglio", "una", "uno", "dei", "delle",
    "per", "da", "di", "su", "con", "e", "o", "il", "lo", "la", "i", "gli", "le", "un", "una",
    "fascia", "media", "medio", "economica", "economico", "top", "migliore", "migliori", "circa",
    "sui", "sul", "euro", "prezzo", "budget"
}

def extract_budget(query: str):
    q = normalize_text(query)
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*€",
        r"(\d+(?:[.,]\d+)?)\s*euro",
        r"sui\s+(\d+(?:[.,]\d+)?)",
        r"entro\s+(\d+(?:[.,]\d+)?)",
        r"max\s+(\d+(?:[.,]\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                pass
    return None

def expand_query_tokens(query: str) -> List[str]:
    base = [t for t in tokenize(query) if t not in STOPWORDS]
    expanded = list(base)

    for token in list(base):
        if token in CATEGORY_SYNONYMS:
            expanded.extend(CATEGORY_SYNONYMS[token])

    for key, vals in CATEGORY_SYNONYMS.items():
        if key in base:
            expanded.extend(vals)

    # euristiche utili
    if "surfcasting" in base:
        expanded.extend(["canna", "canne", "mulinello", "surf"])
    if "trota" in base:
        expanded.extend(["canna", "canne", "trout"])
    if "spinning" in base:
        expanded.extend(["canna", "canne", "mulinello"])
    if "bolognese" in base:
        expanded.extend(["canna", "canne"])
    if "canna" in base or "canne" in base:
        expanded.extend(["rod", "rods"])
    if "mulinello" in base or "mulinelli" in base:
        expanded.extend(["reel", "reels"])

    return unique_preserve([normalize_text(x) for x in expanded if str(x).strip()])

def score_product(row: pd.Series, query: str, expanded_tokens: List[str], budget=None) -> float:
    text = row["all_text"]
    name = row["name_norm"]

    score = 0.0

    # match token
    for tok in expanded_tokens:
        if tok in name:
            score += 10
        elif tok in text:
            score += 4

    # bonus frasi complete
    qn = normalize_text(query)
    if qn and qn in text:
        score += 20

    # bonus combinazioni importanti
    important_pairs = [
        ("canna", "surfcasting"),
        ("mulinello", "surfcasting"),
        ("canna", "trota"),
        ("mulinello", "spinning"),
        ("canna", "bolognese"),
    ]
    for a, b in important_pairs:
        if a in expanded_tokens and b in expanded_tokens and a in text and b in text:
            score += 12

    # brand/serie eventuali
    query_words = tokenize(query)
    for qw in query_words:
        if len(qw) >= 4 and qw in name:
            score += 6

    # budget
    price = row.get("price_num", None)
    if budget is not None and price is not None and not pd.isna(price):
        if price <= budget:
            score += 10
            # vicino al budget meglio
            score += max(0, 8 - abs(budget - price) / max(1, budget) * 10)
        else:
            # piccolo sforamento tollerato ma penalizzato
            if price <= budget * 1.15:
                score += 2
            else:
                score -= 8

    return score

def search_products(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    expanded = expand_query_tokens(query)
    budget = extract_budget(query)

    temp = df.copy()
    temp["score"] = temp.apply(lambda row: score_product(row, query, expanded, budget), axis=1)

    # evita risultati totalmente fuori tema
    must_have_some = [t for t in expanded if t not in {"rod", "rods", "reel", "reels"}]
    filtered_rows = []
    for _, row in temp.iterrows():
        text = row["all_text"]
        hits = sum(1 for t in must_have_some if t in text)
        if hits >= 1:
            filtered_rows.append(True)
        else:
            filtered_rows.append(False)

    temp = temp[filtered_rows].copy()

    # se query contiene canna, escludi mulinelli puri quando possibile
    qn = normalize_text(query)
    if "canna" in qn or "canne" in qn:
        temp = temp[
            temp["all_text"].str.contains("canna|canne|rod", regex=True, na=False)
        ].copy()

    if "mulinello" in qn or "mulinelli" in qn:
        temp = temp[
            temp["all_text"].str.contains("mulinello|mulinelli|reel", regex=True, na=False)
        ].copy()

    if budget is not None and "price_num" in temp.columns:
        under = temp[(temp["price_num"].notna()) & (temp["price_num"] <= budget * 1.15)].copy()
        if not under.empty:
            temp = under

    temp = temp.sort_values(by=["score"], ascending=False)
    temp = temp[temp["score"] > 0]

    return temp.head(top_k).copy()

# =========================================================
# OPENAI
# =========================================================
def ask_openai(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    if client is None:
        return "Al momento il servizio AI non è disponibile."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Al momento non riesco a generare una risposta corretta. Ti consiglio di contattare Emanuele su WhatsApp."

# =========================================================
# RISPOSTE
# =========================================================
def greeting_response() -> str:
    return (
        "Ciao! Sono l’assistente MGFishing 🎣\n\n"
        "Posso aiutarti con:\n"
        "- consigli sui prodotti presenti nel catalogo\n"
        "- spedizioni, tempi, tracking e informazioni utili\n"
        "- meteo pesca, montature e consigli generali di pesca"
    )

def shipping_response(query: str, knowledge: str) -> str:
    if not knowledge.strip():
        return (
            "Per spedizioni, tracking e informazioni ordine ti consiglio di contattare direttamente "
            f"{WHATSAPP_LABEL} su WhatsApp:\n\n{whatsapp_link('Ciao Emanuele, avrei bisogno di informazioni su spedizione o tracking.')}"
        )

    system_prompt = (
        "Sei l'assistente clienti di MGFishing. "
        "Rispondi SOLO usando le informazioni contenute nel testo fornito. "
        "Non inventare nulla. "
        "Se l'informazione non è presente, invita a contattare Emanuele su WhatsApp. "
        "Tono chiaro, professionale e sintetico."
    )

    user_prompt = (
        f"TESTO INFORMATIVO:\n{knowledge}\n\n"
        f"DOMANDA CLIENTE:\n{query}\n\n"
        "Se trovi la risposta nel testo, rispondi direttamente in modo autonomo. "
        "Se non trovi abbastanza informazioni, scrivi di contattare Emanuele su WhatsApp al 3494166335."
    )
    return ask_openai(system_prompt, user_prompt, temperature=0.2)

def product_link_response(query: str) -> str:
    msg = f"Ciao Emanuele, vorrei il link prodotto relativo a: {query}"
    return (
        "Per avere il link corretto del prodotto ti consiglio di scrivere direttamente a "
        f"{WHATSAPP_LABEL} su WhatsApp:\n\n{whatsapp_link(msg)}"
    )

def product_response(query: str, df: pd.DataFrame) -> str:
    if df.empty:
        return (
            "Al momento non riesco a leggere il catalogo prodotti. "
            f"Per un aiuto immediato puoi contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao Emanuele, mi serve un consiglio su un prodotto.')}"
        )

    results = search_products(query, df, top_k=5)

    if results.empty:
        return (
            "Non sono riuscito a trovare nel catalogo prodotti abbastanza pertinenti alla tua richiesta.\n\n"
            f"Per un consiglio più preciso puoi contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao Emanuele, sto cercando questo prodotto: {query}')}"
        )

    catalog_context = []
    for _, row in results.iterrows():
        line = {
            "nome": row.get("name", ""),
            "prezzo": row.get("price", ""),
            "categoria": row.get("category", ""),
            "marca": row.get("brand", ""),
            "descrizione": row.get("description", ""),
        }
        catalog_context.append(line)

    system_prompt = (
        "Sei un assistente esperto di articoli da pesca per MGFishing.\n"
        "Devi consigliare SOLO prodotti presenti nel catalogo fornito.\n"
        "Non devi inventare prodotti, marchi, prezzi o caratteristiche non presenti nei dati.\n"
        "Se ci sono pochi risultati, proponi solo quelli realmente pertinenti.\n"
        "Non parlare di prodotti non presenti nel catalogo.\n"
        "Rispondi in italiano, in modo chiaro, utile e commerciale ma concreto.\n"
        "Niente riferimenti a siti esterni.\n"
        "Non inserire link prodotto."
    )

    user_prompt = (
        f"Richiesta cliente: {query}\n\n"
        f"Prodotti pertinenti trovati nel catalogo:\n{json.dumps(catalog_context, ensure_ascii=False, indent=2)}\n\n"
        "Scrivi una risposta utile e coerente alla richiesta.\n"
        "Se il cliente chiede una fascia di prezzo, tienine conto.\n"
        "Se ci sono 2-4 prodotti pertinenti, spiegali brevemente.\n"
        "Se il match non è perfetto, dillo con onestà ma resta utile."
    )

    return ask_openai(system_prompt, user_prompt, temperature=0.35)

def web_advice_response(query: str) -> str:
    system_prompt = (
        "Sei un assistente esperto di pesca di MGFishing.\n"
        "Rispondi a domande su meteo pesca, montature, tecniche, periodi, esche, terminali e consigli pratici.\n"
        "Non devi citare fonti, siti web o riferimenti esterni.\n"
        "Parla in modo chiaro e utile.\n"
        "Quando si parla di meteo, interpreta SEMPRE la domanda in ottica pesca.\n"
        "Quindi considera vento, mare, onde, pressione, temperatura, acqua, attività del pesce e condizioni pratiche di pesca.\n"
        "Se una risposta dipende da luogo/specie/stagione e manca il dato, dillo chiaramente e dai comunque una linea guida pratica."
    )

    user_prompt = (
        f"Domanda cliente: {query}\n\n"
        "Rispondi come consulente di pesca, senza citare alcun sito o fonte."
    )
    return ask_openai(system_prompt, user_prompt, temperature=0.45)

def generic_response(query: str, df: pd.DataFrame, knowledge: str) -> str:
    # Prova prima catalogo se sembra vagamente prodotto
    maybe_product = search_products(query, df, top_k=3)
    if not maybe_product.empty:
        return product_response(query, df)

    if any(w in normalize_text(query) for w in SHIPPING_WORDS):
        return shipping_response(query, knowledge)

    return web_advice_response(query)

def generate_response(query: str) -> str:
    intent = detect_intent(query)

    if intent == "greeting":
        return greeting_response()
    elif intent == "product_link":
        return product_link_response(query)
    elif intent == "shipping":
        return shipping_response(query, knowledge_text)
    elif intent == "product_advice":
        return product_response(query, catalog_df)
    elif intent == "web_advice":
        return web_advice_response(query)
    else:
        return generic_response(query, catalog_df, knowledge_text)

# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.2rem;
    }
    .small-note {
        color: #7a7a7a;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎣 MGFishing Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Consigli su prodotti MGFishing, spedizioni, montature e meteo pesca</div>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono l’assistente MGFishing 🎣\n\n"
                "Posso aiutarti con consigli sui prodotti, spedizioni, montature e meteo pesca."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Scrivi qui la tua domanda...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sto elaborando la risposta..."):
            answer = generate_response(user_query)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
