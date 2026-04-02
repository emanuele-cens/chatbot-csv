import os
import re
import math
import difflib
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="MGFishing Chatbot",
    page_icon="🎣",
    layout="centered"
)

MODEL_TEXT = os.getenv("OPENAI_MODEL", "gpt-5.4")
WHATSAPP_NUMBER = "393494166335"
WHATSAPP_LINK = f"https://wa.me/{WHATSAPP_NUMBER}?text=Ciao%20Emanuele%2C%20avrei%20bisogno%20di%20assistenza."
CSV_PATH = "prodotti.csv"
KNOWLEDGE_PATH = "knowledge.txt"


# =========================
# OPENAI CLIENT
# =========================
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# =========================
# HELPERS
# =========================
def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\sàèéìòù]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if len(t) > 1]


def contains_any(text: str, keywords: List[str]) -> bool:
    text_n = normalize_text(text)
    return any(k in text_n for k in keywords)


def make_whatsapp_message(custom_text: str = "") -> str:
    if not custom_text:
        custom_text = "Ciao Emanuele, avrei bisogno di assistenza."
    custom_text = custom_text.strip()
    from urllib.parse import quote
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(custom_text)}"


def safe_get(row: Dict, possible_cols: List[str]) -> str:
    for c in possible_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return clean_text(row[c])
    return ""


def detect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_norm = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_norm:
            return cols_norm[c.lower()]
    return ""


# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    except Exception:
        try:
            df = pd.read_csv(CSV_PATH, sep=";", dtype=str, keep_default_na=False)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    df = df.fillna("")

    name_col = detect_column(df, [
        "name", "nome", "title", "titolo", "product", "prodotto",
        "product_name", "nome_prodotto"
    ])
    if not name_col:
        name_col = df.columns[0]

    link_col = detect_column(df, [
        "link", "url", "product_url", "permalink", "href"
    ])

    price_col = detect_column(df, [
        "price", "prezzo", "sale_price", "final_price", "product_price"
    ])

    category_col = detect_column(df, [
        "category", "categoria", "categories", "brand_category", "reparto"
    ])

    brand_col = detect_column(df, [
        "brand", "marca", "manufacturer", "produttore"
    ])

    desc_col = detect_column(df, [
        "description", "descrizione", "short_description", "descrizione_breve"
    ])

    sku_col = detect_column(df, [
        "sku", "reference", "codice", "ean", "id_product"
    ])

    prepared_rows = []
    for _, row in df.iterrows():
        row_dict = {c: clean_text(row[c]) for c in df.columns}

        title = safe_get(row_dict, [name_col]) if name_col else ""
        link = safe_get(row_dict, [link_col]) if link_col else ""
        price = safe_get(row_dict, [price_col]) if price_col else ""
        category = safe_get(row_dict, [category_col]) if category_col else ""
        brand = safe_get(row_dict, [brand_col]) if brand_col else ""
        desc = safe_get(row_dict, [desc_col]) if desc_col else ""
        sku = safe_get(row_dict, [sku_col]) if sku_col else ""

        all_text = " ".join([
            title, category, brand, desc, sku,
            " ".join([clean_text(v) for v in row_dict.values()])
        ])

        prepared_rows.append({
            "_title": title,
            "_link": link,
            "_price": price,
            "_category": category,
            "_brand": brand,
            "_desc": desc,
            "_sku": sku,
            "_search_blob": normalize_text(all_text),
            **row_dict
        })

    return pd.DataFrame(prepared_rows)


@st.cache_data(show_spinner=False)
def load_knowledge() -> str:
    if not os.path.exists(KNOWLEDGE_PATH):
        return ""
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


CATALOG_DF = load_catalog()
KNOWLEDGE_TEXT = load_knowledge()


# =========================
# SEARCH PRODUCTS
# =========================
PRODUCT_HINT_WORDS = [
    "mulinello", "canna", "canna da pesca", "trecciato", "monofilo", "filo",
    "bombarda", "galleggiante", "girella", "amo", "ami", "esca", "esche",
    "popper", "minnow", "jig", "fluorocarbon", "bolognese", "feeder",
    "surfcasting", "spinning", "trota", "carpfishing", "carpa", "serra",
    "palamita", "spigola", "tonnetto", "trabucco", "daiwa", "shimano",
    "colmic", "tubertini", "major craft", "molix", "yuki", "rapture"
]

LINK_HINT_WORDS = [
    "link", "mandami il link", "inviami il link", "url", "pagina prodotto",
    "scheda prodotto", "contatto operatore", "operatore", "whatsapp"
]

KNOWLEDGE_HINT_WORDS = [
    "spedizione", "tracking", "tracciamento", "consegna", "reso", "resi",
    "rimborso", "pagamento", "pagamenti", "contrassegno", "tempi", "ordine",
    "ordini", "assistenza", "costo spedizione", "quanto costa la spedizione"
]

WEB_HINT_WORDS = [
    "meteo", "vento", "mare", "onde", "pioggia", "pressione", "luna",
    "fase lunare", "marea", "maree", "montatura", "montature", "trave",
    "terminali", "bolentino", "surfcasting", "tecnica", "tecniche",
    "come pescare", "consiglio pesca", "periodo migliore"
]


def score_product_row(query: str, row: pd.Series) -> float:
    q_norm = normalize_text(query)
    q_tokens = tokenize(q_norm)
    if not q_tokens:
        return 0.0

    title = normalize_text(row.get("_title", ""))
    category = normalize_text(row.get("_category", ""))
    brand = normalize_text(row.get("_brand", ""))
    desc = normalize_text(row.get("_desc", ""))
    blob = normalize_text(row.get("_search_blob", ""))

    score = 0.0

    for token in q_tokens:
        if token in title:
            score += 4.5
        if token in brand:
            score += 2.5
        if token in category:
            score += 2.2
        if token in desc:
            score += 1.2
        if token in blob:
            score += 0.8

    phrase_ratio = difflib.SequenceMatcher(None, q_norm, title).ratio()
    score += phrase_ratio * 6.0

    # bonus se il titolo è molto simile a parte della query
    short_q = " ".join(q_tokens[:8])
    title_ratio = difflib.SequenceMatcher(None, short_q, title).ratio()
    score += title_ratio * 5.0

    return score


def search_products(query: str, top_k: int = 8) -> pd.DataFrame:
    if CATALOG_DF.empty:
        return pd.DataFrame()

    df = CATALOG_DF.copy()
    df["_score"] = df.apply(lambda r: score_product_row(query, r), axis=1)
    df = df.sort_values("_score", ascending=False)

    # filtro minimo
    df = df[df["_score"] >= 2.4]

    # se query contiene parole prodotto generiche, allarghiamo un po'
    if df.empty and contains_any(query, PRODUCT_HINT_WORDS):
        df = CATALOG_DF.copy()
        df["_score"] = df.apply(lambda r: score_product_row(query, r), axis=1)
        df = df.sort_values("_score", ascending=False).head(top_k)
        df = df[df["_score"] >= 1.6]

    return df.head(top_k)


def catalog_context_from_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""

    rows = []
    for _, row in df.iterrows():
        rows.append(
            "\n".join([
                f"Nome: {clean_text(row.get('_title', ''))}",
                f"Marca: {clean_text(row.get('_brand', ''))}",
                f"Categoria: {clean_text(row.get('_category', ''))}",
                f"Prezzo: {clean_text(row.get('_price', ''))}",
                f"SKU/Codice: {clean_text(row.get('_sku', ''))}",
                f"Descrizione: {clean_text(row.get('_desc', ''))}",
            ])
        )
    return "\n\n---\n\n".join(rows)


# =========================
# INTENT
# =========================
def detect_intent(user_message: str) -> str:
    text = normalize_text(user_message)

    if contains_any(text, LINK_HINT_WORDS):
        return "whatsapp"

    if contains_any(text, KNOWLEDGE_HINT_WORDS):
        return "knowledge"

    if contains_any(text, WEB_HINT_WORDS):
        return "web"

    # se ci sono match forti nel catalogo, trattiamo come prodotto
    matches = search_products(user_message, top_k=5)
    if not matches.empty:
        top_score = float(matches.iloc[0]["_score"])
        if top_score >= 3.2:
            return "product"

    if contains_any(text, PRODUCT_HINT_WORDS):
        return "product"

    return "generic"


# =========================
# AI ANSWERS
# =========================
def ask_model_plain(system_prompt: str, user_prompt: str) -> str:
    client = get_openai_client()
    if client is None:
        return "Manca la variabile ambiente OPENAI_API_KEY."

    try:
        response = client.responses.create(
            model=MODEL_TEXT,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.output_text or "").strip()
    except Exception as e:
        return f"Si è verificato un errore OpenAI: {e}"


def ask_model_with_web(system_prompt: str, user_prompt: str) -> str:
    client = get_openai_client()
    if client is None:
        return "Manca la variabile ambiente OPENAI_API_KEY."

    try:
        response = client.responses.create(
            model=MODEL_TEXT,
            tools=[{"type": "web_search"}],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.output_text or "").strip()
    except Exception as e:
        return f"Si è verificato un errore OpenAI: {e}"


def answer_product(user_message: str) -> str:
    matches = search_products(user_message, top_k=8)

    if matches.empty:
        wa = make_whatsapp_message(
            f"Ciao Emanuele, sto cercando questo prodotto o un consiglio: {user_message}"
        )
        return (
            "Non riesco a individuare con sicurezza il prodotto nel catalogo MGFishing.\n\n"
            f"Per assistenza diretta scrivi a Emanuele su WhatsApp:\n{wa}"
        )

    context = catalog_context_from_df(matches)

    system_prompt = """
Sei l'assistente di MGFishing.
Devi aiutare il cliente SOLO usando i prodotti presenti nel contesto catalogo.
Regole obbligatorie:
- Non inventare mai prodotti non presenti nel contesto.
- Non dire mai che un prodotto esiste se non è nel contesto.
- Se il cliente chiede consigli, proponi solo prodotti presenti nel contesto.
- Se il cliente chiede un link prodotto, NON dare link prodotto: invita a contattare Emanuele su WhatsApp.
- Rispondi in italiano, chiaro, utile e commerciale ma naturale.
- Non citare fonti esterne.
- Se il contesto è poco chiaro, dillo con onestà e rimanda a WhatsApp.
- Se nel contesto ci sono più risultati compatibili, suggerisci i più pertinenti e spiega brevemente perché.
"""

    user_prompt = f"""
Domanda cliente:
{user_message}

Catalogo disponibile:
{context}

Numero WhatsApp Emanuele: 3494166335
Link WhatsApp: {WHATSAPP_LINK}

Genera una risposta utile e precisa.
"""

    answer = ask_model_plain(system_prompt, user_prompt)

    if not answer:
        wa = make_whatsapp_message(
            f"Ciao Emanuele, avrei bisogno di aiuto per questo prodotto: {user_message}"
        )
        return f"Per questo prodotto ti consiglio di contattare Emanuele su WhatsApp:\n{wa}"

    return answer


def answer_knowledge(user_message: str) -> str:
    if not KNOWLEDGE_TEXT.strip():
        wa = make_whatsapp_message(
            f"Ciao Emanuele, avrei bisogno di informazioni su: {user_message}"
        )
        return (
            "Al momento non ho il file knowledge.txt disponibile.\n\n"
            f"Per assistenza scrivi a Emanuele su WhatsApp:\n{wa}"
        )

    system_prompt = """
Sei l'assistente clienti di MGFishing.
Devi rispondere SOLO usando il contenuto del file knowledge fornito.
Regole:
- Non inventare informazioni non presenti nel knowledge.
- Se la risposta è presente nel knowledge, rispondi in modo diretto e autonomo.
- Non rimandare subito a WhatsApp se la risposta è nel knowledge.
- Rimanda a WhatsApp solo se nel knowledge non c'è una risposta sufficiente.
- Rispondi in italiano, in modo chiaro e professionale.
"""

    user_prompt = f"""
Domanda cliente:
{user_message}

Knowledge MGFishing:
{KNOWLEDGE_TEXT}

Numero WhatsApp Emanuele: 3494166335
Link WhatsApp: {WHATSAPP_LINK}
"""

    return ask_model_plain(system_prompt, user_prompt)


def answer_web(user_message: str) -> str:
    system_prompt = """
Sei l'assistente di MGFishing.
Puoi cercare sul web per rispondere a domande su pesca, montature, tecniche, meteo pesca, mare, vento, onde, pressione, luna e consigli pratici.
Regole obbligatorie:
- Rispondi in italiano.
- Non mostrare link, fonti, nomi di siti o riferimenti esterni.
- Se la domanda è sul meteo pesca, interpreta i dati in ottica pesca sportiva.
- Non parlare da meteorologo generico: collega sempre la risposta alla pescabilità quando opportuno.
- Se la domanda è su montature o tecniche, rispondi in modo pratico e concreto.
- Se la domanda chiede di parlare con operatore o è troppo vaga, invita a contattare Emanuele su WhatsApp 3494166335.
"""

    user_prompt = f"""
Rispondi alla seguente domanda del cliente in modo pratico, utile e senza citare siti esterni:

{user_message}
"""

    return ask_model_with_web(system_prompt, user_prompt)


def answer_whatsapp(user_message: str) -> str:
    wa = make_whatsapp_message(f"Ciao Emanuele, avrei bisogno di assistenza su: {user_message}")
    return (
        "Per questo ti consiglio di contattare direttamente Emanuele su WhatsApp:\n\n"
        f"{wa}"
    )


def answer_generic(user_message: str) -> str:
    # prova prima con catalogo se ci sono match almeno discreti
    matches = search_products(user_message, top_k=5)
    if not matches.empty and float(matches.iloc[0]["_score"]) >= 2.7:
        return answer_product(user_message)

    # prova knowledge
    if contains_any(user_message, KNOWLEDGE_HINT_WORDS):
        return answer_knowledge(user_message)

    # fallback web per domande generali pesca
    return answer_web(user_message)


def generate_answer(user_message: str) -> Tuple[str, str]:
    intent = detect_intent(user_message)

    if intent == "whatsapp":
        return answer_whatsapp(user_message), intent

    if intent == "knowledge":
        return answer_knowledge(user_message), intent

    if intent == "web":
        return answer_web(user_message), intent

    if intent == "product":
        return answer_product(user_message), intent

    return answer_generic(user_message), "generic"


# =========================
# UI
# =========================
st.title("🎣 Chatbot MGFishing")
st.caption("Consigli su prodotti, spedizioni, montature e meteo pesca")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ciao! Sono l’assistente MGFishing.\n\n"
                "Posso aiutarti con:\n"
                "- consigli sui prodotti presenti nel catalogo\n"
                "- spedizione, tracking, resi e pagamenti\n"
                "- montature, tecniche e meteo pesca"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Scrivi qui la tua domanda...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Sto cercando la risposta migliore..."):
            answer, intent = generate_answer(user_input)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
