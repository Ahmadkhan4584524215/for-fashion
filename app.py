"""Libas Lab - Pakistani wardrobe stylist and trend radar."""

from __future__ import annotations

import hashlib
import io
import textwrap
from datetime import date
from html import escape

import pandas as pd
import streamlit as st
from PIL import Image

from libas.config import GENDER_OPTIONS, OCCASIONS, SEASON_OPTIONS
from libas.stylist import build_outfits
from libas.trends import (
    AUDIENCES,
    DATA_PATH,
    HORIZON_MONTHS,
    TrendDataError,
    build_trend_table,
    forecast_chart_data,
    load_trend_data,
    top_mover,
    trend_insight,
)
from libas.vision import (
    FashionEncoder,
    ModelLoadError,
    analyze_garments,
    image_to_jpeg,
)

UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; letter-spacing: -0.02em; }

.hero {
    background: linear-gradient(115deg, #12392B 0%, #1F6F52 58%, #2F8A67 100%);
    color: #FBF8F2;
    padding: 2.2rem 2.4rem;
    border-radius: 22px;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -70px;
    width: 260px;
    height: 260px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 50%;
}
.hero .eyebrow {
    letter-spacing: 0.2em;
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #EAD6A4;
}
.hero h1 {
    font-size: 2.9rem;
    margin: 0.25rem 0 0.4rem;
    color: #FBF8F2;
    line-height: 1.05;
}
.hero p {
    max-width: 800px;
    font-size: 1rem;
    color: #EAE4D8;
    margin: 0;
}
.hero .tags {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 1.1rem;
}
.hero .tags span {
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
    color: #F3EDE0;
}
.swatch {
    display: inline-block;
    width: 0.72rem;
    height: 0.72rem;
    border-radius: 50%;
    margin-right: 0.4rem;
    border: 1px solid rgba(0, 0, 0, 0.16);
    vertical-align: -1px;
}
.garment-meta {
    font-size: 0.8rem;
    color: #6D6457;
    margin: 0.2rem 0 0.45rem;
}
[data-testid="stMetric"] {
    background: #F3EEE4;
    border: 1px solid #E4DAC9;
    padding: 0.8rem 1rem;
    border-radius: 16px;
}
[data-testid="stMetricLabel"] { color: #6D6457; }
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 1rem;
}
.kpi-card {
    background: #F3EEE4;
    border: 1px solid #E4DAC9;
    border-radius: 16px;
    padding: 0.9rem 1rem;
}
.kpi-label {
    color: #6D6457;
    font-size: 0.78rem;
    font-weight: 500;
}
.kpi-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.45rem;
    line-height: 1.2;
    color: #23352C;
    margin: 0.28rem 0 0.2rem;
    overflow-wrap: anywhere;
}
.kpi-foot { font-size: 0.78rem; color: #6D6457; }
.kpi-up { color: #1F6F52; font-weight: 600; }
.kpi-down { color: #A6432F; font-weight: 600; }
@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.small-note { color: #6D6457; font-size: 0.86rem; }
</style>
"""

HERO = """
<div class="hero">
    <div class="eyebrow">Pakistan &middot; Men &amp; Women</div>
    <h1>Libas Lab</h1>
    <p>Build outfits from the clothes you already own, finish them with accessories that fit
    a Pakistani occasion, and see which styles are gaining momentum before the season turns.</p>
    <div class="tags">
        <span>FashionCLIP garment vision</span>
        <span>Accessory styling</span>
        <span>Pinterest look ideas</span>
        <span>Six-month trend forecast</span>
    </div>
</div>
"""


@st.cache_resource(show_spinner=False)
def get_encoder() -> FashionEncoder:
    return FashionEncoder()


@st.cache_data(show_spinner=False)
def load_sample_trends() -> pd.DataFrame:
    return load_trend_data()


@st.cache_data(show_spinner=False)
def load_uploaded_trends(payload: bytes) -> pd.DataFrame:
    return load_trend_data(io.BytesIO(payload))


def init_state() -> None:
    if "wardrobe" not in st.session_state:
        st.session_state.wardrobe = {
            gender: {"top": [], "bottom": []} for gender in GENDER_OPTIONS
        }
    if "dismissed" not in st.session_state:
        st.session_state.dismissed = {
            gender: {"top": set(), "bottom": set()} for gender in GENDER_OPTIONS
        }
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = {gender: 0 for gender in GENDER_OPTIONS}
    if "gender" not in st.session_state:
        st.session_state.gender = "Women"
    if "season" not in st.session_state:
        st.session_state.season = (
            "Winter" if date.today().month in {11, 12, 1, 2} else "Summer"
        )
    st.session_state.setdefault("model_status", "")


def _digest(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _load_encoder() -> FashionEncoder | None:
    try:
        with st.spinner("Loading FashionCLIP weights. This happens once."):
            encoder = get_encoder()
        st.session_state.model_status = f"FashionCLIP ready on {encoder.device_label}"
        return encoder
    except ModelLoadError as exc:
        st.session_state.model_status = "error"
        st.error(str(exc))
        return None


def _ingest_uploads(
    files,
    slot: str,
    gender: str,
    store: dict,
    encoder: FashionEncoder,
    known_ids: set[str],
    dismissed_ids: set[str],
) -> int:
    images = []
    metadata = []
    for uploaded in files:
        payload = uploaded.getvalue()
        digest = _digest(payload)
        if digest in known_ids or digest in dismissed_ids:
            continue
        try:
            with Image.open(io.BytesIO(payload)) as raw:
                raw.load()
                image = raw.convert("RGB")
        except Exception:
            st.warning(f"Skipped {uploaded.name}: the file could not be read as an image.")
            continue
        images.append(image)
        metadata.append((digest, uploaded.name, image))
        known_ids.add(digest)

    if not images:
        return 0

    analyses = analyze_garments(images, slot, gender, encoder)
    for (digest, name, image), (attributes, embedding) in zip(metadata, analyses):
        store[slot].append(
            {
                "id": digest,
                "name": name,
                "slot": slot,
                "gender": gender,
                "image": image_to_jpeg(image),
                "embedding": embedding,
                "attrs": attributes,
            }
        )
    return len(analyses)


def _pending_files(files, known_ids: set[str], dismissed_ids: set[str]) -> list:
    pending = []
    for uploaded in files or []:
        digest = _digest(uploaded.getvalue())
        if digest not in known_ids and digest not in dismissed_ids:
            pending.append(uploaded)
    return pending


def _render_item_grid(items: list[dict], slot: str, gender: str) -> None:
    label = "tops" if slot == "top" else "bottoms"
    if not items:
        st.info(f"No {label} yet. Upload a few above to start matching.")
        return
    columns = st.columns(4)
    for index, item in enumerate(items):
        with columns[index % 4]:
            st.image(item["image"], width="stretch")
            attributes = item["attrs"]
            st.markdown(f"**{attributes['type']}**")
            st.markdown(
                '<div class="garment-meta">'
                f'<span class="swatch" style="background:{attributes["color_hex"]}"></span>'
                f"{attributes['color_name']} &middot; {attributes['pattern']} &middot; "
                f"{attributes['formality_label']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{attributes['style']} · {attributes['type_confidence']}% type confidence"
            )
            if st.button(
                "Remove",
                key=f"remove-{gender}-{slot}-{item['id']}",
                width="stretch",
            ):
                items.remove(item)
                st.session_state.dismissed[gender][slot].add(item["id"])
                st.rerun()


def _wardrobe_table(store: dict) -> pd.DataFrame:
    rows = []
    for slot, piece in (("top", "Top"), ("bottom", "Bottom")):
        for item in store[slot]:
            attributes = item["attrs"]
            rows.append(
                {
                    "Item": item["name"],
                    "Piece": piece,
                    "Detected type": attributes["type"],
                    "Colour": attributes["color_name"],
                    "Pattern": attributes["pattern"],
                    "Style": attributes["style"],
                    "Formality": attributes["formality_label"],
                    "Confidence": f"{attributes['type_confidence']}%",
                }
            )
    return pd.DataFrame(rows)


def render_wardrobe(gender: str) -> None:
    store = st.session_state.wardrobe[gender]
    nonce = st.session_state.uploader_nonce[gender]

    st.markdown("## My wardrobe")
    st.caption(
        "Upload one garment per photo, ideally laid flat on a plain background. "
        "FashionCLIP reads the piece and tags its type, pattern, colour and formality. "
        "The cupboard is kept separately for women and men."
    )

    top_column, bottom_column = st.columns(2)
    with top_column:
        top_files = st.file_uploader(
            "Tops — shirts, kurtas, kurtis, blouses",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"upload-top-{gender}-{nonce}",
            help="JPG, PNG or WEBP. Upload several pieces at once.",
        )
    with bottom_column:
        bottom_files = st.file_uploader(
            "Bottoms — trousers, jeans, shalwar, skirts",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"upload-bottom-{gender}-{nonce}",
            help="JPG, PNG or WEBP. Upload several pieces at once.",
        )

    known_ids = {
        item["id"] for slot in ("top", "bottom") for item in store[slot]
    }
    dismissed_top = st.session_state.dismissed[gender]["top"]
    dismissed_bottom = st.session_state.dismissed[gender]["bottom"]
    pending_top = _pending_files(top_files, known_ids, dismissed_top)
    pending_bottom = _pending_files(
        bottom_files, known_ids | {_digest(f.getvalue()) for f in pending_top}, dismissed_bottom
    )

    if pending_top or pending_bottom:
        encoder = _load_encoder()
        if encoder is not None:
            added = 0
            try:
                with st.spinner("FashionCLIP is reading your wardrobe..."):
                    if pending_top:
                        added += _ingest_uploads(
                            pending_top,
                            "top",
                            gender,
                            store,
                            encoder,
                            known_ids,
                            dismissed_top,
                        )
                    if pending_bottom:
                        added += _ingest_uploads(
                            pending_bottom,
                            "bottom",
                            gender,
                            store,
                            encoder,
                            known_ids,
                            dismissed_bottom,
                        )
            except Exception as exc:  # model inference failure should not kill the app
                st.error(f"Garment analysis failed: {exc}")
            else:
                if added:
                    st.success(f"Added {added} garment{'s' if added != 1 else ''} to your cupboard.")

    top_items = store["top"]
    bottom_items = store["bottom"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Tops", len(top_items))
    metric_columns[1].metric("Bottoms", len(bottom_items))
    metric_columns[2].metric(
        "Possible pairings", len(top_items) * len(bottom_items)
    )

    if top_items or bottom_items:
        clear_column, note_column = st.columns([1, 3])
        with clear_column:
            if st.button("Clear this wardrobe", width="stretch"):
                store["top"].clear()
                store["bottom"].clear()
                st.session_state.dismissed[gender] = {"top": set(), "bottom": set()}
                st.session_state.uploader_nonce[gender] += 1
                st.rerun()
        with note_column:
            st.caption(
                "Removing a garment here hides it from the current session; the file stays in the uploader."
            )

    st.markdown("### Tops")
    _render_item_grid(top_items, "top", gender)
    st.markdown("### Bottoms")
    _render_item_grid(bottom_items, "bottom", gender)

    if top_items or bottom_items:
        with st.expander("Cupboard table"):
            st.dataframe(
                _wardrobe_table(store),
                hide_index=True,
                width="stretch",
            )


def _flatten_items(items: list[dict]) -> list[dict]:
    return [
        {**item["attrs"], "id": item["id"], "name": item["name"], "image": item["image"]}
        for item in items
    ]


def render_studio(gender: str, season: str) -> None:
    store = st.session_state.wardrobe[gender]
    st.markdown("## Outfit studio")
    st.caption(
        "Every top and bottom in your cupboard is paired and scored on colour harmony, "
        "occasion formality, pattern balance and Eastern/Western coherence. "
        "Accessories are then matched from a Pakistani catalog."
    )

    if not store["top"] or not store["bottom"]:
        st.info(
            "Upload at least one top and one bottom in **My wardrobe** to generate looks. "
            "FashionCLIP tags the garments, then this tab does the styling."
        )
        return

    controls = st.columns([1.3, 1, 1])
    with controls[0]:
        occasion_label = st.selectbox(
            "Occasion",
            [occasion.label for occasion in OCCASIONS],
            help="The target formality changes the scoring and accessory matches.",
        )
    with controls[1]:
        limit = st.slider("Looks to show", min_value=3, max_value=8, value=5)
    with controls[2]:
        st.metric("Pairings scored", len(store["top"]) * len(store["bottom"]))

    occasion = next(item for item in OCCASIONS if item.label == occasion_label)
    tops = _flatten_items(store["top"])
    bottoms = _flatten_items(store["bottom"])
    all_looks, total = build_outfits(
        tops,
        bottoms,
        gender,
        occasion.key,
        season,
        limit=None,
    )
    looks = all_looks[:limit]

    st.caption(
        f"Showing the top {len(looks)} of {total} pairings for a {season.lower()} "
        f"{occasion.label.lower()} look."
    )

    for rank, look in enumerate(looks, start=1):
        top = look["top"]
        bottom = look["bottom"]
        with st.container(border=True):
            image_columns = st.columns([1, 1, 2.4])
            with image_columns[0]:
                st.image(top["image"], width="stretch")
                st.caption(f"Top · {top['color_name']} {top['type']}")
            with image_columns[1]:
                st.image(bottom["image"], width="stretch")
                st.caption(f"Bottom · {bottom['color_name']} {bottom['type']}")
            with image_columns[2]:
                st.markdown(f"#### Look {rank} · {look['score']}% match")
                st.progress(look["score"], text="Colour, formality and texture balance")
                for reason in look["reasons"]:
                    st.markdown(f"- {reason}")
                st.link_button(
                    "Find this look on Pinterest",
                    look["pinterest_url"],
                    key=f"pinterest-look-{rank}-{top['id'][:8]}-{bottom['id'][:8]}",
                    width="stretch",
                )

            if look["accessories"]:
                st.markdown("**Finish it with**")
                accessory_columns = st.columns(len(look["accessories"]))
                for index, (column, accessory) in enumerate(
                    zip(accessory_columns, look["accessories"])
                ):
                    with column:
                        st.markdown(f"**{accessory['name']}**")
                        st.caption(accessory["group"])
                        st.write(accessory["why"])
                        st.link_button(
                            "Pinterest ideas",
                            accessory["pinterest_url"],
                            key=f"pinterest-accessory-{rank}-{index}-{top['id'][:6]}",
                            width="stretch",
                        )

    with st.expander("All scored combinations"):
        rows = [
            {
                "Rank": index,
                "Top": f"{look['top']['color_name']} {look['top']['type']}",
                "Bottom": f"{look['bottom']['color_name']} {look['bottom']['type']}",
                "Match %": look["score"],
            }
            for index, look in enumerate(all_looks, start=1)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_trends() -> None:
    st.markdown("## Trend radar")
    st.caption(
        "Search-interest data is used as a demand signal: each style is seasonally "
        "forecast six months ahead and ranked by momentum. Men, women and unisex styles "
        "are included."
    )

    with st.expander("Use your own trend data (CSV)"):
        st.markdown(
            "Upload a long-format CSV with **date, keyword, gender, category, interest** "
            "(interest 0–100). A Google Trends export can be reshaped into this format."
        )
        st.code(
            "date,keyword,gender,category,interest\n"
            "2026-01-01,Co-ord sets,Women,Western,42\n"
            "2026-01-01,Linen kurtas,Men,Eastern,31",
            language="csv",
        )
        uploaded = st.file_uploader(
            "Upload trend CSV",
            type=["csv"],
            key="trend-upload",
        )
        if DATA_PATH.exists():
            st.download_button(
                "Download the demo CSV",
                data=DATA_PATH.read_bytes(),
                file_name="pakistan_trends_sample.csv",
                mime="text/csv",
            )

    uploaded_payload = uploaded.getvalue() if uploaded is not None else None

    if uploaded_payload is not None:
        version = hashlib.md5(uploaded_payload).hexdigest()[:8]
        try:
            frame = load_uploaded_trends(uploaded_payload)
            source_label = uploaded.name
            is_demo = False
        except TrendDataError as exc:
            st.error(f"{exc} Falling back to the bundled demo data.")
            try:
                frame = load_sample_trends()
            except TrendDataError as sample_exc:
                st.error(str(sample_exc))
                return
            source_label = "Bundled demo dataset"
            is_demo = True
            version = "demo"
    else:
        try:
            frame = load_sample_trends()
        except TrendDataError as exc:
            st.error(str(exc))
            return
        source_label = "Bundled demo dataset"
        is_demo = True
        version = "demo"

    if is_demo:
        st.info(
            "You are viewing the synthetic demo dataset shipped with the app. "
            "Upload a real trend CSV above to forecast actual market signals."
        )
    st.caption(f"Data source: {source_label} · {len(frame)} observations")

    filters = st.columns([1, 2])
    with filters[0]:
        audience = st.radio("Audience", AUDIENCES, horizontal=True)
    categories = sorted(frame["category"].dropna().unique().tolist())
    with filters[1]:
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=categories,
            key=f"trend-categories-{version}",
        )
    st.caption(
        "Choosing Women or Men also includes unisex styles. "
        f"Forecast horizon: {HORIZON_MONTHS} months from the last observation."
    )

    if not selected_categories:
        st.info("Select at least one category to see the forecast.")
        return

    filtered = frame[frame["category"].isin(selected_categories)]
    table = build_trend_table(filtered, audience=audience)
    if table.empty:
        st.warning(
            "Not enough observations per keyword to fit a forecast. "
            "Provide at least four monthly points for each keyword and gender."
        )
        return

    top_signal = table.iloc[0]
    rising_count = int((table["direction"] == "Rising").sum())
    cooling = top_mover(table, "Cooling")
    last_date = pd.to_datetime(frame["date"]).max()
    window_start = last_date + pd.offsets.MonthBegin(1)
    window_end = window_start + pd.DateOffset(months=HORIZON_MONTHS - 1)

    top_delta = int(top_signal["growth_pct"])
    if cooling is not None:
        cooling_foot = (
            f'<div class="kpi-foot kpi-down">↓ {int(cooling["growth_pct"]):+d}% '
            "year on year</div>"
        )
    else:
        cooling_foot = '<div class="kpi-foot">No style is cooling sharply</div>'
    cooling_value = escape(str(cooling["keyword"])) if cooling is not None else "None flagged"

    kpi_html = textwrap.dedent(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Top rising signal</div>
                <div class="kpi-value">{escape(str(top_signal["keyword"]))}</div>
                <div class="kpi-foot kpi-up">↑ {top_delta:+d}% year on year</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Rising styles</div>
                <div class="kpi-value">{rising_count} of {len(table)}</div>
                <div class="kpi-foot">tracked styles gaining</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Cooling watch-out</div>
                <div class="kpi-value">{cooling_value}</div>
                {cooling_foot}
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Forecast window</div>
                <div class="kpi-value">{HORIZON_MONTHS} months</div>
                <div class="kpi-foot">{window_start:%b %Y} – {window_end:%b %Y}</div>
            </div>
        </div>
        """
    ).strip()
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown("### Forecast detail")
    options = list(table.index)
    selected_index = st.selectbox(
        "Inspect a trend",
        options,
        format_func=lambda index: (
            f"{table.loc[index, 'keyword']} · "
            f"{table.loc[index, 'gender']} · {table.loc[index, 'category']}"
        ),
        label_visibility="collapsed",
    )
    selected_row = table.loc[selected_index]
    chart = forecast_chart_data(
        frame, selected_row["keyword"], selected_row["gender"]
    )
    if not chart.empty:
        st.line_chart(
            chart.set_index("Month")[["Actual", "Forecast"]],
            height=320,
            color=["#1F6F52", "#C9A227"],
        )
        st.caption(
            "Green is observed interest; gold is the seasonal forecast. "
            f"Peak forecast month: {selected_row['peak_month']}. "
            "The model is directional and assumes history repeats its seasonal shape."
        )

    st.markdown("### Ranked trends")
    st.caption(
        "Momentum compares the next three forecast months with the same three months a "
        "year earlier, so a routine seasonal dip is not mistaken for a fading trend."
    )
    display = table[
        [
            "rank",
            "keyword",
            "gender",
            "category",
            "latest",
            "forecast",
            "growth_pct",
            "direction",
            "peak_month",
        ]
    ].rename(
        columns={
            "rank": "Rank",
            "keyword": "Trend",
            "gender": "Audience",
            "category": "Category",
            "latest": "Current",
            "forecast": "Next 3 mo",
            "growth_pct": "Momentum",
            "direction": "Signal",
            "peak_month": "Peak month",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Momentum": st.column_config.NumberColumn(format="%d%%"),
            "Current": st.column_config.NumberColumn(format="%d"),
            "Next 3 mo": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.markdown("### What it means for a Pakistani wardrobe")
    insight_rows = [row for _, row in table.head(3).iterrows()]
    if cooling is not None and cooling.name not in table.head(3).index:
        insight_rows.append(cooling)
    for row in insight_rows:
        with st.container(border=True):
            columns = st.columns([1, 2.4])
            with columns[0]:
                st.markdown(f"**{row['keyword']}**")
                st.caption(
                    f"{row['category']} · {row['gender']} · {row['direction']}"
                )
            with columns[1]:
                st.write(trend_insight(row))

    with st.expander("How the forecast works"):
        st.markdown(
            "Each keyword's monthly search interest is fitted with a **ridge regression "
            "with a linear trend and yearly Fourier seasonality**, then projected six "
            "months forward. Momentum compares the next three forecast months with the "
            "same three months a year earlier, which strips out normal seasonal swings. "
            "It is intentionally lightweight and transparent: no black-box LLM guesses, "
            "and it gets more reliable as you feed it real, longer Google Trends history. "
            "Eid, wedding season and lawn season are captured by the seasonal signal "
            "rather than hard-coded dates."
        )


def main() -> None:
    st.set_page_config(
        page_title="Libas Lab | Pakistani outfit and trend studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Style profile")
        gender = st.radio("Shopping for", GENDER_OPTIONS, key="gender")
        season = st.selectbox("Season", SEASON_OPTIONS, key="season")
        st.caption(
            "The wardrobe and outfit studio follow this profile. The trend radar has "
            "its own audience filter."
        )
        st.divider()
        st.markdown("### Fashion engine")
        status = st.session_state.model_status
        if status == "error":
            st.warning("FashionCLIP did not load. Uploading will retry.")
        elif status:
            st.caption(status)
        else:
            st.caption("FashionCLIP loads on your first upload, then stays cached.")
        st.caption(
            "Model: [patrickjohncyh/fashion-clip]"
            "(https://huggingface.co/patrickjohncyh/fashion-clip) · "
            "first run downloads the weights."
        )
        st.divider()
        st.markdown("### How to get the best results")
        st.caption(
            "Shoot garments flat on a plain background in daylight. One piece per photo. "
            "Makeup-heavy campaign images reduce attribute accuracy."
        )

    wardrobe_tab, studio_tab, trends_tab = st.tabs(
        ["My wardrobe", "Outfit studio", "Trend radar"]
    )
    with wardrobe_tab:
        render_wardrobe(gender)
    with studio_tab:
        render_studio(gender, season)
    with trends_tab:
        render_trends()

    st.divider()
    st.caption(
        "Libas Lab · FashionCLIP for garment recognition · rule-based styling for "
        "explainable matches · seasonal ridge regression for demand forecasts · "
        "Pinterest buttons open a search in a new tab. Trend demo data is synthetic; "
        "forecasts are directional, not guarantees."
    )


if __name__ == "__main__":
    main()
