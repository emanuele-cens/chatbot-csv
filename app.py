import os
import re
import json
import html
import requests
import pandas as pd
import streamlit as st

from typing import List, Dict, Optional
from rapidfuzz import fuzz
from openai import OpenAI


# =========================
# CONFIG
# =========================

APP_TITLE = "MGFishing Chatbot"
MODEL_NAME = "gpt-4.1-mini"
WHATSAPP_NUMBER = "393494166335"
WHATSAPP_CONTACT_NAME = "Emanuele"

CSV_PATH = "prodotti.csv"
KNOWLEDGE_PATH = "knowledge.txt"

REQUEST_TIMEOUT = 20


# =========================
# PAGE
# =========================

st.set_page_config(page_title=APP_TITLE, page_icon="🎣", layout="centered")
st.title("🎣 MGFishing Chatbot")
st.caption("Consigli su prodotti MGFishing, pesca, montature e meteo pesca")


# =========================
# OPENAI
# =========================

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not api_key:
    st.error("OPENAI_API_KEY mancante. Inseriscila nei Secrets di Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
# HELPERS
# =========================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9àèéìòù\s\-\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_str(x) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_whatsapp_link(message: str) -> str:
    from urllib.parse import quote
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(message)}"


def is_link_request(msg: str) -> bool:
    q = normalize_text(msg)
    patterns = [
        "link",
        "mandami il link",
        "mi mandi il link",
        "inviami il link",
        "pagina prodotto",
        "url prodotto",
        "link prodotto",
        "dove lo trovo",
        "aprimi il prodotto",
    ]
    return any(p in q for p in patterns)


def is_operator_request(msg: str) -> bool:
    q = normalize_text(msg)
    patterns = [
        "operatore",
        "parlare con operatore",
        "assistenza",
        "umano",
        "persona vera",
        "contatto",
        "whatsapp",
        "emanuele",
    ]
    return any(p in q for p in patterns)


def is_shipping_request(msg: str) -> bool:
    q = normalize_text(msg)
    patterns = [
        "spedizione",
        "tracking",
        "tracciamento",
        "consegna",
        "ordine",
        "quando arriva",
        "tempi di spedizione",
        "stato ordine",
        "costo spedizione",
        "quanto costa la spedizione",
        "quanto costa spedire",
        "corriere",
    ]
    return any(p in q for p in patterns)


def is_product_request(msg: str) -> bool:
    q = normalize_text(msg)

    product_words = [
        "canna", "mulinello", "trecciato", "monofilo", "bombarda",
        "galleggiante", "amo", "ami", "esca", "artificiale",
        "popper", "jig", "minnow", "spoon", "girella", "piombo",
        "feeder", "pastura", "fluorocarbon", "nylon", "trout",
        "surfcasting", "bolognese", "spinning", "trota", "carpfishing",
        "carp fishing", "bolentino", "ledgering", "mare", "laghetto",
        "consigliami", "consiglio prodotto", "fascia media", "fascia alta",
        "fascia bassa", "economico", "entry level", "top di gamma",
    ]

    non_product_words = [
        "meteo", "vento", "pressione", "luna", "maree", "montatura",
        "terminali", "come fare", "nodo", "assetto", "tecnica", "licenza",
    ]

    if any(w in q for w in product_words):
        return True

    if any(w in q for w in non_product_words):
        return False

    return False


def is_general_fishing_request(msg: str) -> bool:
    q = normalize_text(msg)
    patterns = [
        "meteo", "vento", "pressione", "mare", "onde", "marea", "maree",
        "fase lunare", "luna", "montatura", "montature", "trave",
        "calamento", "terminale", "nodo", "assetto", "bombarda",
        "come pescare", "consigli pesca", "tecnica", "spigola", "serra",
        "orata", "trota", "carpa", "bolognese", "surfcasting", "spinning",
        "bolentino", "feeder", "ledgering",
    ]
    return any(p in q for p in patterns)


def answer_operator() -> str:
    wa = build_whatsapp_link("Ciao Emanuele, avrei bisogno di assistenza.")
    return (
        f"Per assistenza diretta ti consiglio di contattare {WHATSAPP_CONTACT_NAME} su WhatsApp:\n\n"
        f"{wa}"
    )


def fallback_unknown(msg: str) -> str:
    wa = build_whatsapp_link(
        f"Ciao Emanuele, avrei bisogno di aiuto per questa richiesta: {msg}"
    )
    return (
        "Per darti una risposta corretta ti consiglio di contattare direttamente "
        f"{WHATSAPP_CONTACT_NAME} su WhatsApp:\n\n{wa}"
    )


# =========================
# LOAD DATA
# =========================

@st.cache_data(show_spinner=False)
def load_knowledge_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def detect_best_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    existing = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]
    return None


@st.cache_data(show_spinner=False)
def load_catalog(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    if df.empty:
        return df

    name_col = detect_best_column(df, ["name", "product_name", "nome", "title"])
    desc_col = detect_best_column(df, ["description", "descrizione", "short_description", "desc"])
    url_col = detect_best_column(df, ["url", "link", "product_url", "permalink"])
    price_col = detect_best_column(df, ["price", "prezzo", "final_price"])
    category_col = detect_best_column(df, ["category", "categoria", "cat"])
    brand_col = detect_best_column(df, ["brand", "marca", "manufacturer"])
    sku_col = detect_best_column(df, ["sku", "reference", "codice", "ean", "id_product"])
    stock_col = detect_best_column(df, ["stock", "quantity", "qty", "giacenza", "availability"])

    def get_col_value(row, colname):
        if not colname:
            return ""
        return safe_str(row.get(colname, ""))

    rows = []
    for _, row in df.iterrows():
        name = get_col_value(row, name_col)
        description = get_col_value(row, desc_col)
        url = get_col_value(row, url_col)
        price = get_col_value(row, price_col)
        category = get_col_value(row, category_col)
        brand = get_col_value(row, brand_col)
        sku = get_col_value(row, sku_col)
        stock = get_col_value(row, stock_col)

        blob = " ".join([name, description, category, brand, sku, stock]).strip()

        rows.append({
            "name": name,
            "description": description,
            "url": url,
            "price": price,
            "category": category,
            "brand": brand,
            "sku": sku,
            "stock": stock,
            "name_norm": normalize_text(name),
            "search_blob": normalize_text(blob),
        })

    out = pd.DataFrame(rows)

    out = out[(out["name"].astype(str).str.strip() != "")]
    out = out.drop_duplicates(subset=["name", "url"], keep="first").reset_index(drop=True)

    return out


knowledge_text = load_knowledge_text(KNOWLEDGE_PATH)
catalog_df = load_catalog(CSV_PATH)


# =========================
# PRODUCT SEARCH
# =========================

CATEGORY_KEYWORDS = {
    "canna_surfcasting": [
        "canna da surfcasting", "canna surfcasting", "surfcasting", "surf casting"
    ],
    "mulinello_surfcasting": [
        "mulinello da surfcasting", "mulinello surfcasting"
    ],
    "canna_trota": [
        "canna da trota", "canna trota", "trota lago", "tremarella"
    ],
    "mulinello": [
        "mulinello", "reel"
    ],
    "canna": [
        "canna", "rod"
    ],
    "trecciato": [
        "trecciato", "braid"
    ],
    "monofilo": [
        "monofilo", "nylon", "fluorocarbon", "fluoro"
    ],
    "amo": [
        "amo", "ami", "hook", "hooks"
    ],
    "galleggiante": [
        "galleggiante", "float"
    ],
    "bombarda": [
        "bombarda", "bombarde"
    ],
    "pastura": [
        "pastura", "groundbait"
    ],
    "esca": [
        "esca", "artificiale", "hardbait", "softbait", "popper", "minnow", "jig", "spoon"
    ],
}

NEGATIVE_KEYWORDS_BY_CATEGORY = {
    "canna_surfcasting": [
        "ossigenatore", "pesca vivo", "tubo", "pietra", "aeratore"
    ],
    "canna": [
        "ossigenatore", "pesca vivo", "tubo", "pietra", "aeratore"
    ],
    "mulinello_surfcasting": [
        "canna", "rod"
    ],
}


def detect_requested_category(user_msg: str) -> Optional[str]:
    q = normalize_text(user_msg)

    ordered = [
        "canna_surfcasting",
        "mulinello_surfcasting",
        "canna_trota",
        "mulinello",
        "canna",
        "trecciato",
        "monofilo",
        "amo",
        "galleggiante",
        "bombarda",
        "pastura",
        "esca",
    ]

    for cat in ordered:
        for kw in CATEGORY_KEYWORDS.get(cat, []):
            if normalize_text(kw) in q:
                return cat

    return None


def product_matches_category(row: pd.Series, requested_category: Optional[str]) -> bool:
    if not requested_category:
        return True

    text = " ".join([
        safe_str(row.get("name", "")),
        safe_str(row.get("brand", "")),
        safe_str(row.get("category", "")),
        safe_str(row.get("description", "")),
        safe_str(row.get("sku", "")),
    ])
    text = normalize_text(text)

    positive_keywords = CATEGORY_KEYWORDS.get(requested_category, [])
    has_positive = any(normalize_text(k) in text for k in positive_keywords)

    if not has_positive and requested_category == "canna":
        has_positive = "canna" in text
    if not has_positive and requested_category == "mulinello":
        has_positive = "mulinello" in text

    if not has_positive:
        return False

    negative_keywords = NEGATIVE_KEYWORDS_BY_CATEGORY.get(requested_category, [])
    if any(normalize_text(k) in text for k in negative_keywords):
        return False

    return True


def extract_budget_segment(user_msg: str) -> Optional[str]:
    q = normalize_text(user_msg)
    if "fascia media" in q or "medio" in q or "media" in q:
        return "media"
    if "fascia alta" in q or "top di gamma" in q or "alta" in q:
        return "alta"
    if "fascia bassa" in q or "economica" in q or "entry level" in q or "base" in q:
        return "bassa"
    return None


def price_to_float(value: str) -> Optional[float]:
    s = safe_str(value)
    if not s:
        return None

    s = s.replace("€", "").replace("EUR", "").replace("eur", "").strip()
    s = s.replace(".", "").replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def budget_score(price: Optional[float], segment: Optional[str]) -> int:
    if price is None or not segment:
        return 0

    if segment == "bassa":
        if price <= 50:
            return 20
        if price <= 80:
            return 10
        return -10

    if segment == "media":
        if 50 <= price <= 150:
            return 20
        if 35 <= price < 50 or 150 < price <= 180:
            return 10
        return -10

    if segment == "alta":
        if price >= 150:
            return 20
        if price >= 120:
            return 10
        return -10

    return 0


def search_products(query: str, df: pd.DataFrame, limit: int = 5) -> List[Dict]:
    if df.empty:
        return []

    q = normalize_text(query)
    if not q:
        return []

    requested_category = detect_requested_category(query)
    requested_budget = extract_budget_segment(query)

    rows = []
    for _, row in df.iterrows():
        if not product_matches_category(row, requested_category):
            continue

        blob = safe_str(row["search_blob"])
        name_norm = safe_str(row["name_norm"])

        score_name = fuzz.token_set_ratio(q, name_norm)
        score_blob = fuzz.token_set_ratio(q, blob)
        partial_name = fuzz.partial_ratio(q, name_norm)
        partial_blob = fuzz.partial_ratio(q, blob)

        q_words = set(q.split())
        name_words = set(name_norm.split())
        common = len(q_words.intersection(name_words))

        price_val = price_to_float(row.get("price", ""))
        b_score = budget_score(price_val, requested_budget)

        final_score = max(score_name, score_blob, partial_name, partial_blob) + (common * 5) + b_score

        rows.append({
            "score": final_score,
            "name": safe_str(row["name"]),
            "url": safe_str(row["url"]),
            "price": safe_str(row["price"]),
            "brand": safe_str(row["brand"]),
            "category": safe_str(row["category"]),
            "description": safe_str(row["description"]),
            "stock": safe_str(row["stock"]),
            "sku": safe_str(row["sku"]),
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    threshold = 62 if requested_category else 55
    filtered = [r for r in rows if r["score"] >= threshold][:limit]

    return filtered


def build_products_context(products: List[Dict]) -> str:
    blocks = []
    for i, p in enumerate(products, start=1):
        blocks.append(
            f"""Prodotto {i}
Nome: {p.get('name', '')}
Brand: {p.get('brand', '')}
Categoria: {p.get('category', '')}
Prezzo: {p.get('price', '')}
Disponibilità: {p.get('stock', '')}
Descrizione: {p.get('description', '')}
URL interno: {p.get('url', '')}
SKU: {p.get('sku', '')}
"""
        )
    return "\n".join(blocks).strip()


# =========================
# SIMPLE WEB SEARCH
# =========================

def ddg_search_snippets(query: str, max_results: int = 5) -> List[Dict]:
    """
    Ricerca semplice via DuckDuckGo HTML.
    Serve solo per recuperare testo di supporto.
    Non mostriamo mai fonti all'utente finale.
    """
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.post(
            url,
            data={"q": query},
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        html_text = resp.text

        results = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>.*?<a class="result__snippet".*?>(.*?)</a>',
            re.S
        )

        for match in pattern.finditer(html_text):
            href = re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
            title = re.sub(r"<[^>]+>", " ", html.unescape(match.group(2)))
            snippet = re.sub(r"<[^>]+>", " ", html.unescape(match.group(3)))

            title = re.sub(r"\s+", " ", title).strip()
            snippet = re.sub(r"\s+", " ", snippet).strip()

            if title or snippet:
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": href,
                })

            if len(results) >= max_results:
                break

        return results
    except Exception:
        return []


def build_web_context_for_fishing(query: str) -> str:
    enriched_query = (
        f"{query} pesca meteo pesca montature terminali condizioni pesca "
        f"vento mare pressione pesci consigli pratici"
    )
    snippets = ddg_search_snippets(enriched_query, max_results=5)

    if not snippets:
        return ""

    chunks = []
    for i, s in enumerate(snippets, start=1):
        chunks.append(
            f"Fonte web {i}\nTitolo: {s['title']}\nSnippet: {s['snippet']}"
        )
    return "\n\n".join(chunks)


# =========================
# OPENAI ANSWERS
# =========================

def call_openai_text(system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            temperature=0.2,
            max_output_tokens=700,
        )
        return (response.output_text or "").strip()
    except Exception as e:
        return f"Errore temporaneo nella generazione della risposta: {e}"


def shipping_answer(msg: str, knowledge: str) -> str:
    system_prompt = f"""
Sei l'assistente clienti di MGFishing.
Rispondi sempre in italiano.
Devi usare SOLO le informazioni presenti nella knowledge aziendale qui sotto.

Regole:
- Rispondi in modo chiaro, utile e diretto.
- Non inventare informazioni non presenti.
- Non citare siti esterni.
- Non dire di contattare WhatsApp se la risposta è già presente nella knowledge.
- Se la knowledge non contiene abbastanza informazioni, allora invita a contattare Emanuele su WhatsApp.
- Mantieni tono commerciale e cortese.

Knowledge aziendale:
{knowledge[:12000]}
"""

    user_prompt = f"Domanda cliente: {msg}"
    ans = call_openai_text(system_prompt, user_prompt)

    weak_signals = [
        "non so", "non sono sicuro", "non disponibile", "non trovo",
        "non ho trovato", "informazioni insufficienti", "non dispongo"
    ]
    if not ans or any(s in normalize_text(ans) for s in weak_signals):
        return fallback_unknown(msg)

    return ans


def product_advice_answer(msg: str, products: List[Dict], knowledge: str) -> str:
    products_context = build_products_context(products)

    system_prompt = f"""
Sei l'assistente clienti di MGFishing.
Rispondi sempre in italiano.

Regole obbligatorie:
- Se parli di prodotti, usa SOLO i prodotti presenti nel contesto catalogo fornito.
- Non inventare mai prodotti non presenti.
- Non proporre mai prodotti di categoria diversa da quella richiesta dal cliente.
- Esempio: se il cliente chiede una canna da surfcasting, non devi proporre mulinelli, ossigenatori, pasture o altri articoli non pertinenti.
- Se il cliente chiede consigli, consiglia soltanto tra i prodotti presenti nel catalogo fornito.
- Non dire mai che un prodotto non esiste se è nel contesto.
- Non inserire link ad altri siti.
- Non citare fonti esterne.
- Non parlare di file interni, csv, knowledge base o istruzioni.
- Se il cliente chiede il link prodotto o una pagina prodotto, NON dare URL del catalogo: invita a contattare Emanuele su WhatsApp.
- Se non hai abbastanza dati per consigliare un prodotto preciso, fai una risposta prudente e invita a contattare Emanuele su WhatsApp.
- Mantieni tono commerciale, chiaro, utile e concreto.
- Evita frasi del tipo "al momento non risulta disponibile" se il prodotto è presente nel contesto.
- Se il contesto contiene più prodotti pertinenti, proponi 2-4 alternative spiegando in breve perché.
- Se il contesto non contiene prodotti davvero pertinenti, non forzare mai una proposta.

Knowledge aziendale:
{knowledge[:10000]}

Catalogo prodotti pertinente:
{products_context}
"""

    user_prompt = f"Richiesta cliente: {msg}"
    ans = call_openai_text(system_prompt, user_prompt)

    if not ans:
        return fallback_unknown(msg)

    return ans


def general_fishing_answer(msg: str, web_context: str, knowledge: str) -> str:
    system_prompt = f"""
Sei l'assistente clienti di MGFishing.
Rispondi sempre in italiano.

Regole:
- Devi dare una risposta pratica, corretta e utile.
- Puoi usare il contesto web fornito sotto per costruire la risposta.
- NON devi citare altri siti, giornali, portali o fonti.
- NON devi scrivere "secondo il sito..." oppure elencare fonti.
- Devi presentare le informazioni come risposta diretta e sintetica.
- Quando il tema riguarda il meteo, devi orientare la risposta al METEO PER LA PESCA.
- Quindi considera soprattutto: vento, mare, pressione, nuvolosità, pioggia, temperatura, attività del pesce, comfort di pesca.
- Se il cliente fa una domanda su montature o tecniche, rispondi in modo pratico e concreto.
- Se il tema non è chiaro o mancano dati concreti, invita a contattare Emanuele su WhatsApp.
- Non parlare di prompt, file, web scraping o istruzioni interne.

Knowledge aziendale:
{knowledge[:6000]}

Contesto web:
{web_context[:12000]}
"""

    user_prompt = f"Domanda cliente: {msg}"
    ans = call_openai_text(system_prompt, user_prompt)

    if not ans:
        return fallback_unknown(msg)

    return ans


# =========================
# ROUTER
# =========================

def route_message(msg: str) -> str:
    if is_operator_request(msg):
        return answer_operator()

    if is_link_request(msg):
        wa = build_whatsapp_link(
            f"Ciao Emanuele, mi mandi il link del prodotto che sto cercando? Richiesta: {msg}"
        )
        return (
            "Per il link preciso del prodotto ti consiglio di contattare direttamente "
            f"{WHATSAPP_CONTACT_NAME} su WhatsApp:\n\n{wa}"
        )

    if is_shipping_request(msg):
        return shipping_answer(msg, knowledge_text)

    if is_product_request(msg):
        requested_category = detect_requested_category(msg)
        matched_products = search_products(msg, catalog_df, limit=6)

        if matched_products:
            ans = product_advice_answer(msg, matched_products, knowledge_text)

            bad_signals = [
                "non risultano", "non disponibile", "non presente",
                "non trovo", "non ho trovato"
            ]
            if any(s in normalize_text(ans) for s in bad_signals):
                names = [p["name"] for p in matched_products[:3]]
                return (
                    "Ho trovato questi prodotti pertinenti nel catalogo MGFishing:\n\n- " +
                    "\n- ".join(names) +
                    "\n\nSe vuoi, dimmi quale ti interessa e ti consiglio quello più adatto."
                )
            return ans

        if requested_category:
            if requested_category == "canna_surfcasting":
                wa = build_whatsapp_link(
                    "Ciao Emanuele, sto cercando una canna da surfcasting fascia media. Mi puoi consigliare un modello disponibile?"
                )
                return (
                    "Non sono riuscito a individuare con precisione nel catalogo una canna da surfcasting adatta alla tua richiesta.\n\n"
                    "Per un consiglio corretto e immediato ti consiglio di contattare direttamente Emanuele su WhatsApp:\n\n"
                    f"{wa}"
                )

            wa = build_whatsapp_link(
                f"Ciao Emanuele, avrei bisogno di un consiglio prodotto per: {msg}"
            )
            return (
                "Non sono riuscito a individuare con precisione un prodotto adatto nel catalogo per questa richiesta.\n\n"
                "Per evitare di indicarti articoli sbagliati, ti consiglio di contattare direttamente Emanuele su WhatsApp:\n\n"
                f"{wa}"
            )

    if is_general_fishing_request(msg):
        web_context = build_web_context_for_fishing(msg)
        ans = general_fishing_answer(msg, web_context, knowledge_text)

        weak_signals = [
            "non so", "non sono sicuro", "non ho abbastanza elementi",
            "non posso verificare", "mi manca il contesto"
        ]
        if not ans or any(w in normalize_text(ans) for w in weak_signals):
            return fallback_unknown(msg)

        return ans

    return fallback_unknown(msg)


# =========================
# UI CHAT
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ciao! Sono l'assistente MGFishing. Posso aiutarti con consigli sui prodotti, spedizioni, montature e meteo pesca."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Scrivi qui la tua domanda...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sto rispondendo..."):
            response_text = route_message(prompt)
            st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
