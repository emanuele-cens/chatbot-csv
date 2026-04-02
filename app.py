import os
import re
import io
import csv
import json
import unicodedata
from typing import List

import pandas as pd
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

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================================================
# UTILS
# =========================================================
def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " e ")
    text = re.sub(r"[^a-z0-9àèéìòù\s€.,:/_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9€]+", normalize_text(text)) if len(t) > 1]


def unique_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_float(value):
    if value is None:
        return None
    s = str(value).strip().replace("€", "").replace("eur", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s.lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def whatsapp_link(message: str) -> str:
    from urllib.parse import quote
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(message)}"


def contains_any(text: str, words) -> bool:
    t = normalize_text(text)
    return any(normalize_text(w) in t for w in words)


# =========================================================
# LOAD FILES
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

    best_df = None
    best_cols = 0

    for sep in [",", ";", "\t", "|"]:
        try:
            df = try_read_csv_with_sep(content, sep)
            if df is not None and len(df.columns) > best_cols:
                best_df = df
                best_cols = len(df.columns)
        except Exception:
            pass

    if best_df is None:
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

    best_df.columns = [str(c).strip() for c in best_df.columns]
    best_df = best_df.fillna("")

    def find_col(possibles):
        cols = list(best_df.columns)
        for p in possibles:
            for c in cols:
                if c.lower().strip() == p.lower().strip():
                    return c
        for p in possibles:
            for c in cols:
                if p.lower().strip() in c.lower().strip():
                    return c
        return None

    col_name = find_col(["name", "nome", "title", "titolo", "product_name", "nome prodotto"])
    col_desc = find_col(["description", "descrizione", "short_description", "desc"])
    col_price = find_col(["price", "prezzo", "final_price", "prezzo finale", "prezzo iva incl"])
    col_url = find_col(["url", "link", "product_url", "permalink", "href"])
    col_cat = find_col(["category", "categoria", "cat"])
    col_brand = find_col(["brand", "marca", "manufacturer"])

    if not col_name and len(best_df.columns) > 0:
        col_name = best_df.columns[0]

    std = pd.DataFrame()
    std["name"] = best_df[col_name] if col_name in best_df.columns else ""
    std["description"] = best_df[col_desc] if col_desc in best_df.columns else ""
    std["price"] = best_df[col_price] if col_price in best_df.columns else ""
    std["url"] = best_df[col_url] if col_url in best_df.columns else ""
    std["category"] = best_df[col_cat] if col_cat in best_df.columns else ""
    std["brand"] = best_df[col_brand] if col_brand in best_df.columns else ""

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

    std = std[std["name_norm"].str.len() > 2].copy()
    std = std.drop_duplicates(subset=["name_norm"]).reset_index(drop=True)
    return std


@st.cache_data(show_spinner=False)
def load_knowledge(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


catalog_df = load_catalog(CSV_PATH)
knowledge_text = load_knowledge(KNOWLEDGE_PATH)

# =========================================================
# KEYWORDS / INTENT
# =========================================================
GREETING_WORDS = {
    "ciao", "salve", "buongiorno", "buonasera", "hey"
}

LINK_WORDS = {
    "link", "mandami il link", "inviami il link", "url", "pagina prodotto",
    "scheda prodotto", "apri prodotto", "dove lo trovo", "dammi il link"
}

PRODUCT_HINTS = {
    "canna", "canne", "mulinello", "mulinelli", "trecciato", "monofilo", "fluorocarbon",
    "bombarda", "galleggiante", "guadino", "artificiale", "artificiali", "esca", "esche",
    "jig", "popper", "minnow", "feeder", "surfcasting", "spinning", "trota", "bolognese",
    "eging", "bolentino", "ledgering", "carpfishing", "ami", "amo", "girelle", "piombi",
    "kit", "combo"
}

STORE_INFO_WORDS = {
    "spedizione", "spedizioni", "tracking", "tracciamento", "consegna", "consegne",
    "ordine", "stato ordine", "tempi di spedizione", "resi", "reso",
    "pagamento", "pagamenti", "pagare", "come posso pagare", "metodi di pagamento",
    "contrassegno", "carta", "paypal", "bonifico", "assistenza", "whatsapp",
    "tempo di consegna", "quanto costa la spedizione", "costo spedizione"
}

WEB_ADVICE_WORDS = {
    "meteo", "vento", "mare", "onde", "marea", "pressione", "pioggia", "temperatura",
    "montatura", "montature", "trave", "finale", "terminale", "terminali", "innesco",
    "come pescare", "quando pescare", "pesca", "orario migliore", "luna", "corrente",
    "spiaggia", "scaduta", "acqua velata", "acqua torbida", "mareggiata"
}

STOPWORDS = {
    "mi", "puoi", "potresti", "vorrei", "voglio", "consigliami", "consiglia", "consiglio",
    "una", "uno", "dei", "delle", "per", "da", "di", "su", "con", "e", "o", "il", "lo",
    "la", "i", "gli", "le", "un", "nel", "nella", "dello", "della", "del", "delle",
    "fascia", "media", "medio", "economica", "economico", "top", "migliore", "migliori",
    "circa", "sui", "sul", "euro", "prezzo", "budget", "adatta", "adatto", "cerco", "cercando"
}


def detect_intent(query: str) -> str:
    q = normalize_text(query)

    if q in GREETING_WORDS:
        return "greeting"

    if contains_any(q, LINK_WORDS):
        return "product_link"

    if contains_any(q, STORE_INFO_WORDS):
        return "store_info"

    if contains_any(q, WEB_ADVICE_WORDS):
        return "web_advice"

    if contains_any(q, PRODUCT_HINTS):
        return "product_advice"

    return "generic"


# =========================================================
# PRODUCT SEARCH
# =========================================================
def extract_budget(query: str):
    q = normalize_text(query)
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*€",
        r"(\d+(?:[.,]\d+)?)\s*euro",
        r"entro\s+(\d+(?:[.,]\d+)?)",
        r"sui\s+(\d+(?:[.,]\d+)?)",
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
    toks = [t for t in tokenize(query) if t not in STOPWORDS]
    out = list(toks)

    synonyms = {
        "trota": ["trout", "area", "tremarella"],
        "surfcasting": ["surf", "beach", "ledgering"],
        "spinning": ["spin"],
        "bolognese": ["bolo"],
        "canna": ["canne", "rod"],
        "canne": ["canna", "rod"],
        "mulinello": ["mulinelli", "reel"],
        "mulinelli": ["mulinello", "reel"],
        "artificiale": ["artificiali", "minnow", "popper", "jig"],
        "artificiali": ["artificiale", "minnow", "popper", "jig"],
    }

    for t in list(toks):
        out.extend(synonyms.get(t, []))

    return unique_preserve(out)


def score_product(row: pd.Series, query: str, tokens: List[str], budget=None) -> float:
    name = row["name_norm"]
    text = row["all_text"]
    score = 0.0

    for tok in tokens:
        if tok in name:
            score += 10
        elif tok in text:
            score += 4

    qn = normalize_text(query)
    if qn and qn in text:
        score += 20

    price = row.get("price_num")
    if budget is not None and price is not None and pd.notna(price):
        if price <= budget:
            score += 8
        elif price <= budget * 1.10:
            score += 2
        else:
            score -= 6

    if "canna" in qn or "canne" in qn:
        if re.search(r"\bcanna\b|\bcanne\b|\brod\b", text):
            score += 10
        else:
            score -= 20

    if "mulinello" in qn or "mulinelli" in qn:
        if re.search(r"\bmulinello\b|\bmulinelli\b|\breel\b", text):
            score += 10
        else:
            score -= 20

    return score


def search_products(query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tokens = expand_query_tokens(query)
    budget = extract_budget(query)

    temp = df.copy()
    temp["score"] = temp.apply(lambda row: score_product(row, query, tokens, budget), axis=1)

    qn = normalize_text(query)

    if "canna" in qn or "canne" in qn:
        temp = temp[temp["all_text"].str.contains(r"\bcanna\b|\bcanne\b|\brod\b", regex=True, na=False)].copy()

    if "mulinello" in qn or "mulinelli" in qn:
        temp = temp[temp["all_text"].str.contains(r"\bmulinello\b|\bmulinelli\b|\breel\b", regex=True, na=False)].copy()

    if budget is not None:
        under = temp[(temp["price_num"].notna()) & (temp["price_num"] <= budget * 1.10)].copy()
        if not under.empty:
            temp = under

    temp = temp.sort_values(by="score", ascending=False)
    temp = temp[temp["score"] > 0]

    return temp.head(top_k).copy()


# =========================================================
# OPENAI
# =========================================================
def ask_openai(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
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
        return "Al momento non riesco a generare una risposta corretta."


# =========================================================
# RESPONSES
# =========================================================
def greeting_response() -> str:
    return (
        "Ciao! Sono l’assistente MGFishing 🎣\n\n"
        "Posso aiutarti con:\n"
        "- consigli sui prodotti del catalogo\n"
        "- spedizioni, pagamenti, tracking e info negozio\n"
        "- meteo pesca, montature e consigli di pesca"
    )


def product_link_response(query: str) -> str:
    msg = f"Ciao Emanuele, vorrei il link del prodotto: {query}"
    return (
        f"Per avere il link corretto del prodotto ti consiglio di scrivere direttamente a {WHATSAPP_LABEL} su WhatsApp:\n\n"
        f"{whatsapp_link(msg)}"
    )


def store_info_response(query: str, knowledge: str) -> str:
    if not knowledge.strip():
        return (
            f"Per assistenza diretta ti consiglio di contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao Emanuele, avrei bisogno di informazioni sul negozio.')}"
        )

    system_prompt = (
        "Sei l'assistente clienti di MGFishing.\n"
        "Rispondi usando SOLO le informazioni presenti nel testo fornito.\n"
        "La domanda può essere formulata in modi diversi ma con lo stesso significato, ad esempio:\n"
        "- 'metodi di pagamento'\n"
        "- 'come posso pagare?'\n"
        "- 'che pagamento accettate?'\n"
        "Quindi devi riconoscere anche le parafrasi.\n"
        "Se il testo contiene la risposta, rispondi in modo diretto, naturale e autonomo.\n"
        "Se il testo NON contiene la risposta, invita a contattare Emanuele su WhatsApp al 3494166335.\n"
        "Non inventare nulla."
    )

    user_prompt = (
        f"TESTO KNOWLEDGE:\n{knowledge}\n\n"
        f"DOMANDA UTENTE:\n{query}"
    )

    reply = ask_openai(system_prompt, user_prompt, temperature=0.1)

    if not reply or "non riesco" in normalize_text(reply):
        return (
            f"Per questa informazione ti consiglio di contattare direttamente {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao Emanuele, avrei bisogno di informazioni su: {query}')}"
        )

    return reply


def product_response(query: str, df: pd.DataFrame) -> str:
    if df.empty:
        return (
            "Al momento non riesco a leggere il catalogo prodotti.\n\n"
            f"Per un aiuto immediato puoi contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link('Ciao Emanuele, mi serve un consiglio su un prodotto.')}"
        )

    results = search_products(query, df, top_k=5)

    if results.empty:
        return (
            "Non sono riuscito a trovare nel catalogo prodotti davvero pertinenti alla tua richiesta.\n\n"
            f"Per un consiglio più preciso ti consiglio di contattare {WHATSAPP_LABEL} su WhatsApp:\n\n"
            f"{whatsapp_link(f'Ciao Emanuele, sto cercando questo prodotto: {query}')}"
        )

    catalog_context = []
    for _, row in results.iterrows():
        catalog_context.append({
            "nome": row.get("name", ""),
            "prezzo": row.get("price", ""),
            "categoria": row.get("category", ""),
            "marca": row.get("brand", ""),
            "descrizione": row.get("description", ""),
        })

    system_prompt = (
        "Sei un assistente esperto di pesca per MGFishing.\n"
        "Devi consigliare SOLO prodotti presenti nel catalogo fornito.\n"
        "Non inventare prodotti, categorie, marchi, prezzi o caratteristiche.\n"
        "Se la richiesta parla di canne, non consigliare mulinelli o altri accessori salvo richiesta esplicita.\n"
        "Se la richiesta parla di mulinelli, non consigliare canne o altri accessori salvo richiesta esplicita.\n"
        "Se i risultati sono pochi o non perfetti, dillo con onestà.\n"
        "Niente link prodotto.\n"
        "Niente riferimenti esterni.\n"
        "Risposta chiara, pratica e commerciale."
    )

    user_prompt = (
        f"Richiesta cliente: {query}\n\n"
        f"Prodotti trovati nel catalogo:\n{json.dumps(catalog_context, ensure_ascii=False, indent=2)}\n\n"
        "Rispondi consigliando solo i prodotti davvero coerenti con la richiesta."
    )

    return ask_openai(system_prompt, user_prompt, temperature=0.25)


def web_advice_response(query: str) -> str:
    system_prompt = (
        "Sei un assistente esperto di pesca di MGFishing.\n"
        "Rispondi a domande su montature, meteo pesca, tecniche, terminali, inneschi, spot, orari e consigli pratici.\n"
        "Quando si parla di meteo, interpreta SEMPRE la domanda in ottica pesca.\n"
        "Quindi considera vento, mare, onde, pressione, corrente, attività del pesce e condizioni pratiche.\n"
        "Non citare siti esterni o fonti esterne.\n"
        "Se mancano dati come zona, specie o stagione, dillo e dai comunque una linea guida utile."
    )

    user_prompt = f"Domanda cliente: {query}"
    return ask_openai(system_prompt, user_prompt, temperature=0.4)


def generic_response(query: str, df: pd.DataFrame, knowledge: str) -> str:
    qn = normalize_text(query)

    if contains_any(qn, STORE_INFO_WORDS):
        return store_info_response(query, knowledge)

    maybe_products = search_products(query, df, top_k=3)
    if not maybe_products.empty and contains_any(qn, PRODUCT_HINTS):
        return product_response(query, df)

    return web_advice_response(query)


def generate_response(query: str) -> str:
    intent = detect_intent(query)

    if intent == "greeting":
        return greeting_response()
    if intent == "product_link":
        return product_link_response(query)
    if intent == "store_info":
        return store_info_response(query, knowledge_text)
    if intent == "product_advice":
        return product_response(query, catalog_df)
    if intent == "web_advice":
        return web_advice_response(query)

    return generic_response(query, catalog_df, knowledge_text)


# =========================================================
# UI
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎣 MGFishing Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Consigli su prodotti MGFishing, pagamenti, spedizioni, montature e meteo pesca</div>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono l’assistente MGFishing 🎣\n\n"
                "Posso aiutarti con consigli sui prodotti, info su pagamenti e spedizioni, "
                "montature e meteo pesca."
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
