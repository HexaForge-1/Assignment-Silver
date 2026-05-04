"""
SQLite + Pandas dashboards with **Gen‑AI narration**
(OpenAI-compatible API).
"""

from __future__ import annotations

import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

# Dev containers (github Codespaces) have no display; Agg keeps matplotlib stable for st.pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")


import json

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from scipy import stats
from sqlalchemy import create_engine, text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Demo data lives on disk ---
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
DATABASE_FILE_PATH = DATA_FOLDER / "shop.sqlite"

# --- Lightweight sentiment analyzer ---
mood_meter = SentimentIntensityAnalyzer()


def _run_sql_script(database_engine, batch_sql_script: str) -> None:
    with database_engine.begin() as connection:
        for raw_chunk in batch_sql_script.split(";"):
            cleaned_chunk = raw_chunk.strip()
            if cleaned_chunk:
                connection.execute(text(cleaned_chunk))


def _empty_every_table(database_engine):
    with database_engine.begin() as connection:
        table_names = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        )
        for (table_label,) in table_names:
            connection.execute(text(f'DROP TABLE IF EXISTS "{table_label}"'))


def _bootstrap(database_engine) -> None:
    _run_sql_script(
        database_engine,
        """
        CREATE TABLE products (id INT PRIMARY KEY, sku TEXT, name TEXT, category TEXT, list_price REAL);
        CREATE TABLE inventory (product_id INT PRIMARY KEY, qty INT, reorder INT, lead_days INT);
        CREATE TABLE sales (id INTEGER PRIMARY KEY AUTOINCREMENT, sale_date TEXT, product_id INT, units INT, revenue REAL, channel TEXT);
        CREATE TABLE feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, comment TEXT);
        CREATE TABLE campaigns (name TEXT, channel TEXT, spend REAL, revenue REAL);
        CREATE TABLE competitors (brand TEXT, category TEXT, price REAL, promo TEXT);
        CREATE TABLE trends (week TEXT, index_score REAL);
        """,
    )
    demo_dice = random.Random(42)
    catalogue_rows_for_demo = [
        (1, "P1", "Dark Roast", "Coffee", 16.49),
        (2, "P2", "Organic Blend", "Coffee", 9.99),
        (3, "P3", "Mug", "Merch", 12.99),
    ]
    with database_engine.begin() as connection:
        for sku_row in catalogue_rows_for_demo:
            connection.execute(
                text("INSERT INTO products VALUES (:i,:sku,:n,:cat,:p)"),
                dict(i=sku_row[0], sku=sku_row[1], n=sku_row[2], cat=sku_row[3], p=sku_row[4]),
            )
        connection.execute(text("INSERT INTO inventory VALUES (1,18,25,7),(2,80,30,5),(3,120,20,14)"))
        for _ in range(55):
            comment_timestamp = datetime(2025, 3, 1) + timedelta(days=demo_dice.randint(0, 39))
            connection.execute(
                text("INSERT INTO feedback VALUES (NULL,:t,:m)"),
                dict(
                    t=comment_timestamp.isoformat(),
                    m=demo_dice.choice(
                        ["Love the coffee.", "Slow shipping.", "Great staff.", "Too expensive.", "Will order again."]
                    ),
                ),
            )
        newest_day_in_demo = date(2025, 5, 1)
        for day_offset in range(90):
            calendar_day = newest_day_in_demo - timedelta(days=day_offset)
            weekend_or_calendar_quirk = (
                0.6 if calendar_day.weekday() == 6 else (0.75 if 25 <= calendar_day.day <= 30 else 1.0)
            )
            product_that_sold = demo_dice.choice([1, 1, 2, 3])
            units_that_sold = demo_dice.choice([1, 2, 3])
            shelf_price_that_day = {1: 16.49, 2: 9.99, 3: 12.99}[product_that_sold]
            connection.execute(
                text("INSERT INTO sales VALUES (NULL,:d,:pid,:u,:rev,:ch)"),
                dict(
                    d=calendar_day.isoformat(),
                    pid=product_that_sold,
                    u=units_that_sold,
                    rev=round(
                        shelf_price_that_day * units_that_sold * weekend_or_calendar_quirk * demo_dice.uniform(0.95, 1.05),
                        2,
                    ),
                    ch=demo_dice.choice(["Web", "Store", "Partner"]),
                ),
            )
        for flyer_name, marketing_channel, cash_out, modeled_return in [
            ("Spring mail", "Email", 800, 5200),
            ("Meta ads", "Social", 2400, 6800),
            ("Partner push", "Partner", 500, 2100),
        ]:
            connection.execute(
                text("INSERT INTO campaigns VALUES (:n,:ch,:s,:r)"),
                dict(n=flyer_name, ch=marketing_channel, s=cash_out, r=modeled_return),
            )
        for rival_name, section, sticker_price in [
            ("BeanCo", "Coffee", 11.5),
            ("QuickMart", "Coffee", 8.9),
            ("GiftCo", "Merch", 11.0),
        ]:
            connection.execute(
                text("INSERT INTO competitors VALUES (:b,:c,:p,'')"),
                dict(b=rival_name, c=section, p=sticker_price),
            )
        week_cursor = date(2025, 3, 3)
        for _ in range(8):
            demand_reading = round(60 + demo_dice.uniform(-4, 4), 1)
            connection.execute(
                text("INSERT INTO trends VALUES (:w,:v)"),
                dict(w=week_cursor.isoformat(), v=demand_reading),
            )
            week_cursor += timedelta(days=7)


def engine():
    DATA_FOLDER.mkdir(exist_ok=True)
    if not DATABASE_FILE_PATH.exists():
        database_engine = create_engine(f"sqlite:///{DATABASE_FILE_PATH}")
        _bootstrap(database_engine)
        return database_engine
    return create_engine(f"sqlite:///{DATABASE_FILE_PATH}")


def reseed():
    DATA_FOLDER.mkdir(exist_ok=True)
    if DATABASE_FILE_PATH.exists():
        try:
            DATABASE_FILE_PATH.unlink()
        except PermissionError:
            database_engine = create_engine(f"sqlite:///{DATABASE_FILE_PATH}")
            _empty_every_table(database_engine)
            _bootstrap(database_engine)
            return database_engine
    database_engine = create_engine(f"sqlite:///{DATABASE_FILE_PATH}")
    _bootstrap(database_engine)
    return database_engine


def read_sql(sql_question: str, database_engine, **knobs) -> pd.DataFrame:
    with database_engine.connect() as session:
        return pd.read_sql_query(text(sql_question), session, params=knobs)


# --- light analytics ---


def sales_monthly(database_engine):
    return read_sql(
        "SELECT strftime('%Y-%m',sale_date) ym, SUM(revenue) revenue FROM sales GROUP BY 1 ORDER BY 1", database_engine
    )


def sales_story(database_engine):
    revenue_each_calendar_month = sales_monthly(database_engine)
    if revenue_each_calendar_month.empty:
        return {}, ["No sales yet — load data first."]
    newest_month_tag = revenue_each_calendar_month["ym"].iloc[-1]
    comparison_month_tag = (
        revenue_each_calendar_month["ym"].iloc[-2] if len(revenue_each_calendar_month) > 1 else newest_month_tag
    )
    newest_month_total = float(
        revenue_each_calendar_month.loc[
            revenue_each_calendar_month["ym"] == newest_month_tag, "revenue"
        ].iloc[0]
    )
    earlier_month_total = float(
        revenue_each_calendar_month.loc[
            revenue_each_calendar_month["ym"] == comparison_month_tag, "revenue"
        ].iloc[0]
    )
    month_over_month_change_pct = (
        0 if earlier_month_total == 0 else (newest_month_total - earlier_month_total) / earlier_month_total * 100
    )

    bestselling_rows = read_sql(
        """
        SELECT p.name n, SUM(s.revenue) r FROM sales s JOIN products p ON p.id=s.product_id
        WHERE strftime('%Y-%m',s.sale_date)=:ym GROUP BY p.name ORDER BY r DESC LIMIT 3
        """,
        database_engine,
        ym=newest_month_tag,
    )

    coaching_lines = []
    if month_over_month_change_pct < -5:
        coaching_lines.append(
            f"Revenue is down about {abs(month_over_month_change_pct):.0f}% vs the prior month — check promos or stock-outs."
        )
    if month_over_month_change_pct > 5:
        coaching_lines.append("Revenue rose — double down on what worked last month.")
    for _, favourite_row in bestselling_rows.iterrows():
        coaching_lines.append(f"{favourite_row['n']} drove ${favourite_row['r']:.0f} — keep it in stock.")
    if not coaching_lines:
        coaching_lines.append("Numbers look steady; keep weekly eyes on top SKUs.")

    daily_slices_for_newest_month = read_sql(
        "SELECT sale_date d, SUM(revenue) revenue FROM sales WHERE strftime('%Y-%m',sale_date)=:ym GROUP BY sale_date",
        database_engine,
        ym=newest_month_tag,
    )
    weekend_to_weekday_ratio = 1.0
    if len(daily_slices_for_newest_month) > 5:
        daily_slices_for_newest_month["weekday_index"] = pd.to_datetime(
            daily_slices_for_newest_month["d"]
        ).dt.weekday
        weekday_ticket = daily_slices_for_newest_month.loc[
            daily_slices_for_newest_month["weekday_index"] < 5, "revenue"
        ].mean()
        weekend_ticket = daily_slices_for_newest_month.loc[
            daily_slices_for_newest_month["weekday_index"] >= 5, "revenue"
        ].mean()
        weekend_to_weekday_ratio = (
            float(weekend_ticket / weekday_ticket) if weekday_ticket else 1.0
        )
        if weekend_to_weekday_ratio < 0.7:
            coaching_lines.append("Weekends trail weekdays — try a small Saturday bundle.")

    headline_numbers = dict(
        newest_month=newest_month_tag,
        comparison_month=comparison_month_tag,
        newest_revenue_total=newest_month_total,
        earlier_revenue_total=earlier_month_total,
        month_over_month_change_pct=month_over_month_change_pct,
        weekend_to_weekday_ratio=weekend_to_weekday_ratio,
        tops=bestselling_rows,
    )
    return headline_numbers, coaching_lines


def sentiment_block(database_engine):
    feedback_rows = read_sql("SELECT comment FROM feedback", database_engine)
    if feedback_rows.empty:
        return feedback_rows.assign(compound=0.0), "No comments yet."
    labelled = feedback_rows.copy()
    labelled["compound"] = labelled["comment"].fillna("").map(
        lambda wording: mood_meter.polarity_scores(wording)["compound"]
    )
    cheers_share = (labelled["compound"] >= 0.05).mean()
    grim_share = (labelled["compound"] <= -0.05).mean()
    meh_share = 1 - cheers_share - grim_share
    takeaway_sentence = (
        f"Average tone {labelled['compound'].mean():+.2f} "
        f"(pos {cheers_share:.0%}, neutral {meh_share:.0%}, neg {grim_share:.0%})."
    )
    return labelled, takeaway_sentence


def sentiment_vs_sales(database_engine):
    comments_by_day = read_sql("SELECT date(created_at) d, comment FROM feedback", database_engine)
    register_totals_by_day = read_sql("SELECT sale_date d, SUM(revenue) revenue FROM sales GROUP BY sale_date", database_engine)
    if comments_by_day.empty or register_totals_by_day.empty:
        return None, "Need both feedback and sales for correlation."

    comments_by_day["calendar_day"] = pd.to_datetime(comments_by_day["d"])
    comments_by_day["daily_mood"] = comments_by_day["comment"].fillna("").map(
        lambda wording: mood_meter.polarity_scores(wording)["compound"]
    )
    mood_per_calendar_day = comments_by_day.groupby("calendar_day", as_index=False)["daily_mood"].mean()

    register_totals_by_day["calendar_day"] = pd.to_datetime(register_totals_by_day["d"])

    moods_joined_with_tills = register_totals_by_day.merge(mood_per_calendar_day, on="calendar_day", how="inner")
    if len(moods_joined_with_tills) < 8:
        return None, "Not enough overlapping days."

    correlation_strength, statistic_confidence_hint = stats.pearsonr(
        moods_joined_with_tills["daily_mood"], moods_joined_with_tills["revenue"]
    )
    explainer_sentence = (
        f"Sentiment vs revenue same-day: r≈{correlation_strength:.2f} (p={statistic_confidence_hint:.2f})."
    )
    return float(correlation_strength), explainer_sentence


def competitors(database_engine):
    shelf_price_sheet = read_sql(
        """
        SELECT c.brand, c.category, c.price comp_price,
               (SELECT AVG(list_price) FROM products p WHERE p.category=c.category) our_avg
        FROM competitors c
        """,
        database_engine,
    )
    friendly_nudges = []
    if not shelf_price_sheet.empty:
        shelf_price_sheet["price_gap"] = shelf_price_sheet["comp_price"] - shelf_price_sheet["our_avg"]
        if not shelf_price_sheet[shelf_price_sheet["price_gap"] < -1].empty:
            friendly_nudges.append("Rivals beat you on shelf price in spots — bundles or loyalty soften that.")
        if not shelf_price_sheet[shelf_price_sheet["price_gap"] > 1].empty:
            friendly_nudges.append("You skew higher than some peers — say why (quality, freshness, speed).")
    if not friendly_nudges:
        friendly_nudges.append("Pricing is clustered — differentiate with story and reliability.")
    return shelf_price_sheet, friendly_nudges


def inventory(database_engine):
    stock_sheet = read_sql(
        """
        SELECT p.sku, p.name, i.qty, i.reorder, i.lead_days,
               COALESCE((SELECT SUM(units)/30.0 FROM sales s WHERE s.product_id=p.id AND s.sale_date>=date('now','-30 day')),0) spd
        FROM inventory i JOIN products p ON p.id=i.product_id
        """,
        database_engine,
    )
    stockroom_reminders = []
    for _, item_row in stock_sheet.iterrows():
        days_on_hand_estimate = (
            item_row["qty"] / item_row["spd"] if item_row["spd"] > 0 else 999
        )
        if item_row["qty"] <= item_row["reorder"]:
            stockroom_reminders.append(f"{item_row['sku']} is at/below reorder — place an order.")
        elif (
            days_on_hand_estimate < item_row["lead_days"]
            and item_row["spd"] > 0
        ):
            stockroom_reminders.append(
                f"{item_row['sku']} may run short before the next delivery lands."
            )
    if stock_sheet["spd"].sum() == 0:
        stockroom_reminders.append("Little recent movement — double-check forecasts before buying more.")
    if not stockroom_reminders:
        stockroom_reminders.append("Stock looks reasonable for recent sell-through.")
    return stock_sheet, stockroom_reminders


def marketing(database_engine):
    campaign_performance = read_sql("SELECT * FROM campaigns", database_engine)
    campaign_performance["rough_return_multiple"] = campaign_performance.apply(
        lambda spreadsheet_row: spreadsheet_row["revenue"] / spreadsheet_row["spend"]
        if spreadsheet_row["spend"]
        else 0,
        axis=1,
    )
    playbook_lines = []
    rising_star_row = campaign_performance.sort_values("rough_return_multiple", ascending=False).head(1)
    laggard_row = campaign_performance.sort_values("rough_return_multiple").head(1)
    if not rising_star_row.empty:
        playbook_lines.append(
            f"'{rising_star_row.iloc[0]['name']}' looks strongest — consider nudging budget there."
        )
    if not laggard_row.empty and laggard_row.iloc[0]["rough_return_multiple"] < 1.1:
        playbook_lines.append(
            f"Pause or retarget '{laggard_row.iloc[0]['name']}' until performance firms up."
        )
    if not playbook_lines:
        playbook_lines.append("Campaign mix is fine — review weekly.")
    return campaign_performance, playbook_lines


def trends(database_engine):
    return read_sql("SELECT week w, index_score v FROM trends ORDER BY week", database_engine)


def sqlite_cache_fingerprint() -> str:
    if not DATABASE_FILE_PATH.exists():
        return "none"
    stt = DATABASE_FILE_PATH.stat()
    return f"{stt.st_mtime_ns}:{stt.st_size}"


def _openai_api_key_and_base_url() -> tuple[str | None, str | None]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets["OPENAI_API_KEY"]).strip()  # type: ignore[index]
        except Exception:
            key = ""
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    base_url = base if base else None
    return (key if key else None), base_url


def _choose_model_name() -> str:
    raw = os.environ.get("OPENAI_MODEL", "").strip()
    return raw if raw else "gpt-4o-mini"


def _missing_key_message() -> str:
    return (
        "**Gen‑AI is not configured.** Set **`OPENAI_API_KEY`** in the environment, or "
        "create `.streamlit/secrets.toml` with `OPENAI_API_KEY = \"sk-…\"`, then rerun."
    )


def _card_keys() -> list[str]:
    return [
        "pulse_markdown",
        "competitor_markdown",
        "customer_voice_markdown",
        "ops_marketing_markdown",
    ]


def _invoke_chat(messages: list[dict[str, str]], *, temperature: float, json_mode: bool) -> str:
    api_key, base_url = _openai_api_key_and_base_url()
    if not api_key:
        raise RuntimeError("missing API key")

    kw: dict[str, object] = {"api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    client = OpenAI(**kw)
    kwargs: dict[str, object] = {
        "model": _choose_model_name(),
        "temperature": temperature,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def build_evidence_bundle_plain(database_engine) -> str:
    """Plain-text FACTS ledger for grounding (retrieve-then-generate pattern)."""
    headline_numbers, coaching_lines = sales_story(database_engine)
    moods_table, tone_sentence = sentiment_block(database_engine)
    rho_like, corr_sentence = sentiment_vs_sales(database_engine)
    competitor_frame, comp_hints = competitors(database_engine)
    stock_sheet, stock_hints = inventory(database_engine)
    campaign_view, ads_hints = marketing(database_engine)
    demand_frame = trends(database_engine)

    try:
        sample_lines = moods_table["comment"].dropna().astype(str).head(45).tolist()
    except Exception:
        sample_lines = []

    chunks: list[str] = []
    chunks.append("# NUMERIC HEADLINE (trusted)")
    for key, val in headline_numbers.items():
        if isinstance(val, pd.DataFrame):
            chunks.append(f"### {key}\n{val.to_string(max_rows=12)}")
        else:
            chunks.append(f"- {key}: {val}")
    chunks.append("\n## Code-generated hints")
    for line in coaching_lines[:14]:
        chunks.append(f"- {line}")
    chunks.append("\n## Customer summaries")
    chunks.append(tone_sentence)
    chunks.append("" if rho_like is None else corr_sentence)
    chunks.append("\n## Raw comment samples")
    chunks.extend([f"- {snippet}" for snippet in sample_lines])
    chunks.append("\n## Competitor rows")
    chunks.append(competitor_frame.head(25).to_string())
    chunks.append("| hints: " + " · ".join(comp_hints[:6]))
    chunks.append("\n## Inventory snapshot")
    chunks.append(stock_sheet.to_string())
    chunks.append("| hints: " + " · ".join(stock_hints[:8]))
    chunks.append("\n## Campaign ROI table")
    chunks.append(campaign_view.to_string())
    chunks.append("| hints: " + " · ".join(ads_hints[:6]))
    chunks.append("\n## External trend index rows")
    chunks.append(demand_frame.to_string())
    joined = "\n".join(part for part in chunks if part is not None)
    return joined[:115_000]


_DASH_RULES = (
    "You are a pragmatic Gen-AI business copilot helping a SMALL shop owner. "
    "You receive authoritative FACTS from their database. Explain clearly. "
    "Use ONLY facts present — do not hallucinate SKU-level revenue if missing.\n\n"
    "Return STRICT JSON only with these string fields (Markdown inside strings is allowed):\n"
    '{\n  "pulse_markdown": "sales & pacing insights",\n'
    '  "competitor_markdown": "competitive/market-price guidance",\n'
    '  "customer_voice_markdown": "tone, themes from comments, linkage to FACTS sentiment stats",\n'
    '  "ops_marketing_markdown": "inventory + merchandising plus campaign optimisation"\n}'
)


@st.cache_data(ttl=300)
def genai_dashboard_markdown_cards(fingerprint_snap: str) -> dict[str, str]:
    eng = engine()
    ledger = build_evidence_bundle_plain(eng)
    if not ledger.strip():
        return {k: _missing_key_message() for k in _card_keys()}

    api_key, _ = _openai_api_key_and_base_url()
    if not api_key:
        placeholder = {_k: _missing_key_message() for _k in _card_keys()}
        return placeholder

    try:
        raw = _invoke_chat(
            [
                {"role": "system", "content": _DASH_RULES},
                {"role": "user", "content": f"FACTS:\n{ledger}\n\nProduce the JSON payload now."},
            ],
            temperature=0.35,
            json_mode=True,
        )
        blob = json.loads(raw)
        return {
            "pulse_markdown": str(blob.get("pulse_markdown", "")).strip() or "**(Empty model reply)**",
            "competitor_markdown": str(blob.get("competitor_markdown", "")).strip() or "**(Empty model reply)**",
            "customer_voice_markdown": str(blob.get("customer_voice_markdown", "")).strip()
            or "**(Empty model reply)**",
            "ops_marketing_markdown": str(blob.get("ops_marketing_markdown", "")).strip()
            or "**(Empty model reply)**",
        }
    except Exception as exc:
        msg = f"**Gen‑AI synthesis failed.** ({exc})\nSupply a valid `{_choose_model_name()}` credential and retry."
        return {k: msg for k in _card_keys()}


_CHAT_SYSTEM = (
    "You answer as a concise Gen‑AI SMB analyst. Anchor every numeric statement to the FACTS block. "
    "If FACTS omit data, admit it. Offer prioritized next actions (inventory, CX, promotions, staffing). "
    "Answer in fluent Markdown bullets."
)


@st.cache_data(ttl=240)
def genai_answer_owner_question(question: str, fingerprint_snap: str) -> str:
    q = question.strip()
    ledger = build_evidence_bundle_plain(engine())

    api_key, _ = _openai_api_key_and_base_url()
    if not api_key:
        return _missing_key_message()

    payload = (
        f"OWNER QUESTION:\n{q}\n\nFACTS_LEDGER:\n{ledger}\n\n"
        "Respond with decisive but honest guidance referencing only supported evidence."
    )
    try:
        return _invoke_chat(
            [{"role": "system", "content": _CHAT_SYSTEM}, {"role": "user", "content": payload}],
            temperature=0.3,
            json_mode=False,
        )
    except Exception as exc:
        return f"**Gen‑AI call failed.** {exc}"


_REPORT_SYSTEM = (
    "Write a Markdown business briefing for an owner labelled with the cadence they requested. "
    "Include sections: Highlights, Risks/opportunities, Customer voice themes, Competitive moves, Inventory & fulfilment, "
    "Marketing next steps. Facts must originate from FACTS only; mark gaps explicitly."
)


@st.cache_data(ttl=320)
def genai_periodic_report_markdown(report_label: str, fingerprint_snap: str) -> str:
    ledger = build_evidence_bundle_plain(engine())

    api_key, _ = _openai_api_key_and_base_url()
    if not api_key:
        return _missing_key_message()

    heading = (
        f"Produce a polished `{report_label}` Business Intelligence recap as Markdown headings + bullets "
        "(no hallucinated KPIs)."
    )
    payload = heading + "\n\nFACTS:\n" + ledger
    try:
        return _invoke_chat(
            [
                {"role": "system", "content": _REPORT_SYSTEM},
                {"role": "user", "content": payload},
            ],
            temperature=0.25,
            json_mode=False,
        )
    except Exception as exc:
        return f"**Report generation failed.** {exc}"


def answer_question(shop_question: str, database_engine) -> str:
    """Gen‑AI Q&A layered on deterministic FACTS (database_engine retained for callers)."""
    del database_engine
    cleaned = shop_question.strip()
    if not cleaned:
        return "**Type a concrete question.** Example: “Why might revenue lag between periods?”."
    fingerprint = sqlite_cache_fingerprint()
    return genai_answer_owner_question(cleaned, fingerprint)


def monthly_report(database_engine, label: str = "Weekly") -> str:
    """Gen-AI written digest (database_engine arg kept API-compatible; fingerprint uses file)."""
    return genai_periodic_report_markdown(label, sqlite_cache_fingerprint())


# --- Streamlit UI ---
st.set_page_config(page_title="Shop BI (one file)", layout="wide")
st.title("Local business helper")
st.caption(
    "SQLite + Pandas dashboards, Plotly & Matplotlib, VADER mood histogram · **GPT-style narratives** grounded "
    "on your numbers (needs **`OPENAI_API_KEY`** env or `.streamlit/secrets.toml`)."
)

if st.sidebar.button("Reload sample data"):
    reseed()
    st.cache_data.clear()
    st.sidebar.success("Demo data refreshed.")

shop_database = engine()
facts_fingerprint = sqlite_cache_fingerprint()
genai_cards = genai_dashboard_markdown_cards(facts_fingerprint)
if _openai_api_key_and_base_url()[0]:
    st.sidebar.success("Gen‑AI (OpenAI-compatible) credentials detected.")
else:
    st.sidebar.warning("Set **`OPENAI_API_KEY`** — Gen‑AI text will pause until configured.")

sales_tab, ask_tab, market_tab, mood_tab, operations_tab, report_tab = st.tabs(
    ["Sales", "Ask me", "Competitors & trends", "Sentiment", "Inventory & marketing", "Report"]
)

with sales_tab:
    revenue_each_month = sales_monthly(shop_database)
    if revenue_each_month.empty:
        st.warning("No sales rows.")
    else:
        st.plotly_chart(
            px.line(revenue_each_month, x="ym", y="revenue", markers=True, title="Revenue by month"),
            use_container_width=True,
        )
        recent_slices = revenue_each_month.tail(6).reset_index(drop=True)
        bar_chapter, axes_for_bars = plt.subplots(figsize=(6.2, 2.9))
        axes_for_bars.bar(range(len(recent_slices)), recent_slices["revenue"])
        axes_for_bars.set_xticks(range(len(recent_slices)))
        axes_for_bars.set_xticklabels(recent_slices["ym"], rotation=35, ha="right")
        axes_for_bars.set_title("Matplotlib bars (same recent months)")
        st.pyplot(bar_chapter, clear_figure=True)
    headline_snapshot, _ = sales_story(shop_database)
    if headline_snapshot:
        col_latest_month, col_previous_month, col_shift = st.columns(3)
        col_latest_month.metric("Latest month", f"${headline_snapshot['newest_revenue_total']:,.0f}")
        col_previous_month.metric("Prior month", f"${headline_snapshot['earlier_revenue_total']:,.0f}")
        col_shift.metric("Change", f"{headline_snapshot['month_over_month_change_pct']:+.1f}%")
    st.subheader("Gen‑AI read on pulse & SKU focus")
    st.markdown(genai_cards.get("pulse_markdown", _missing_key_message()))

with ask_tab:
    # Keep answers in session state.
    if "bi_last_reply_markdown" not in st.session_state:
        st.session_state.bi_last_reply_markdown = None
    if "bi_question_box_nonce" not in st.session_state:
        st.session_state.bi_question_box_nonce = 0

    st.caption(
        "Write your question, then press **Answer**. The box clears for the next question "
        "so typing stays left-to-right reliably."
    )

    question_widget_key = f"bi_owner_question_{st.session_state.bi_question_box_nonce}"

    with st.form("owner_question_form"):
        owners_question_box = st.text_area(
            "Your question",
            placeholder='Example: "Why did sales drop last month?"',
            height=96,
            key=question_widget_key,
            label_visibility="collapsed",
        )
        question_submitted = st.form_submit_button("Answer")

    if question_submitted:
        trimmed_question = owners_question_box.strip()
        if trimmed_question:
            st.session_state.bi_last_reply_markdown = answer_question(trimmed_question, shop_database)
            st.session_state.bi_question_box_nonce += 1
            st.rerun()
        else:
            st.info("Please type something first.")

    if st.session_state.bi_last_reply_markdown:
        st.divider()
        st.markdown("#### Latest answer")
        st.markdown(st.session_state.bi_last_reply_markdown)

with market_tab:
    competitor_table, _ = competitors(shop_database)
    st.dataframe(competitor_table, use_container_width=True)
    st.subheader("Gen‑AI competitive briefing")
    st.markdown(genai_cards.get("competitor_markdown", _missing_key_message()))
    demand_story = trends(shop_database)
    if not demand_story.empty:
        st.plotly_chart(
            px.line(demand_story, x="w", y="v", title="Demand index (demo)"),
            use_container_width=True,
        )

with mood_tab:
    moods_table, moods_sentence = sentiment_block(shop_database)
    st.write(moods_sentence)
    mood_money_correlation, stats_sentence = sentiment_vs_sales(shop_database)
    st.caption(
        stats_sentence
        if mood_money_correlation is None
        else f"{stats_sentence} (informative only on small demos.)"
    )
    st.plotly_chart(
        px.histogram(moods_table, x="compound", nbins=14, title="Comment moods"),
        use_container_width=True,
    )
    st.subheader("Gen‑AI customer-voice synthesis")
    st.markdown(genai_cards.get("customer_voice_markdown", _missing_key_message()))

with operations_tab:
    stocking_table, _ = inventory(shop_database)
    st.subheader("Inventory")
    st.dataframe(stocking_table, use_container_width=True)

    campaigns_table, _ = marketing(shop_database)
    st.subheader("Campaigns")
    st.dataframe(campaigns_table, use_container_width=True)
    st.plotly_chart(
        px.scatter(
            campaigns_table,
            x="spend",
            y="revenue",
            size="revenue",
            hover_name="name",
            title="Spend vs revenue",
        ),
        use_container_width=True,
    )
    st.subheader("Gen‑AI inventory & campaign guidance")
    st.markdown(genai_cards.get("ops_marketing_markdown", _missing_key_message()))

with report_tab:
    cadence_you_want_shown = st.radio("Label", ["Weekly", "Monthly"], horizontal=True)
    st.markdown(monthly_report(shop_database, label=cadence_you_want_shown))
