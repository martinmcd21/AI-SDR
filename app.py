# app.py
# PowerDash AI SDR – Outbound Email Sender (Founder-controlled)
#
# Features included:
# - Upload MULTIPLE Apollo CSVs (solves 25-export limitation)
# - Campaign selector: Scheduler / Suite
# - One template editor (subject + body) applied to all selected leads
# - Save templates per campaign (templates.json) + reset to defaults
# - Optional: “Generate suggested template with AI” (one click, then you edit)
# - Select which leads to send to (checkbox column)
# - Dry run mode (default ON)
# - Send-window scheduling (Now / Today 8 / Today 11 / Today 2 / Tomorrow 8)
# - SMTP send with delay throttling
# - Run log + downloadable CSV of “sent/processed” records

import os
import json
import time
import smtplib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# OpenAI is optional in this app (only used for template suggestions)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# Config / Secrets
# -----------------------------
APP_TZ_DEFAULT = "Europe/London"
TEMPLATES_FILE = "templates.json"

FROM_NAME_DEFAULT = "Martin McDonald"
DEFAULT_SEND_DELAY_SECONDS = 120  # conservative

# Prefer Streamlit secrets if present; fall back to env vars
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if "secrets" in dir(st) and key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)


SMTP_HOST = get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(get_secret("SMTP_PORT", "587") or "587")
SMTP_USER = get_secret("SMTP_USER")
SMTP_PASSWORD = get_secret("SMTP_PASSWORD")

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------
# Default templates
# -----------------------------
DEFAULT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Scheduler": {
        "subject": "Simplifying interview scheduling",
        "body": (
            "Hi {first_name},\n\n"
            "Interview scheduling often creates unnecessary back-and-forth, especially when coordinating availability with hiring managers.\n\n"
            "I’m building PowerDash Interview Scheduler as a lightweight option: hiring managers share a screenshot or PDF of availability, and candidates select a suitable time themselves — no calendar integrations.\n\n"
            "Worth a quick look?\n\n"
            "Best,\n"
            "Martin\n"
            "Martin McDonald\n"
            "PowerDash HR"
        ),
    },
    "Suite": {
        "subject": "Reducing HR admin overhead",
        "body": (
            "Hi {first_name},\n\n"
            "Many HR teams still spend a lot of time on manual operational work across hiring, onboarding, and reporting.\n\n"
            "I’m building PowerDash HR Suite as a lightweight operational layer that sits alongside existing HR systems and helps reduce that manual effort.\n\n"
            "Open to a brief look if useful?\n\n"
            "Best,\n"
            "Martin\n"
            "Martin McDonald\n"
            "PowerDash HR"
        ),
    },
}


# -----------------------------
# Helpers: templates persistence
# -----------------------------
def load_templates() -> Dict[str, Dict[str, str]]:
    # Start with defaults, then overlay saved values
    templates = json.loads(json.dumps(DEFAULT_TEMPLATES))
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k in templates.keys():
                if isinstance(saved, dict) and k in saved and isinstance(saved[k], dict):
                    templates[k]["subject"] = str(saved[k].get("subject", templates[k]["subject"]))
                    templates[k]["body"] = str(saved[k].get("body", templates[k]["body"]))
        except Exception:
            # If file is corrupted, ignore and keep defaults
            pass
    return templates


def save_templates(templates: Dict[str, Dict[str, str]]) -> None:
    with open(TEMPLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)


# -----------------------------
# Helpers: CSV normalization
# -----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_leads(df: pd.DataFrame) -> pd.DataFrame:
    # Apollo exports can vary slightly; try sensible fallbacks.
    col_first = pick_col(df, ["First Name", "first_name", "First"])
    col_last = pick_col(df, ["Last Name", "last_name", "Last"])
    col_title = pick_col(df, ["Title", "Job Title", "job_title"])
    col_company = pick_col(df, ["Company Name", "Company", "company_name"])
    col_email = pick_col(df, ["Email", "email", "Work Email", "Primary Email"])

    missing = [("First Name", col_first), ("Title", col_title), ("Company Name", col_company), ("Email", col_email)]
    missing_required = [label for label, col in missing if col is None]
    if missing_required:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(missing_required)
            + ". Please export with First Name, Title, Company Name, Email."
        )

    out = pd.DataFrame(
        {
            "first_name": df[col_first].fillna("").astype(str),
            "last_name": df[col_last].fillna("").astype(str) if col_last else "",
            "job_title": df[col_title].fillna("").astype(str),
            "company": df[col_company].fillna("").astype(str),
            "email": df[col_email].fillna("").astype(str),
        }
    )

    # Basic cleaning
    out["first_name"] = out["first_name"].str.strip()
    out["job_title"] = out["job_title"].str.strip()
    out["company"] = out["company"].str.strip()
    out["email"] = out["email"].str.strip().str.lower()

    # Drop rows missing email
    out = out[out["email"].astype(bool)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# -----------------------------
# Helpers: OpenAI (template suggestion only)
# -----------------------------
AI_SYSTEM_PROMPT = """
You write short, founder-led outbound email templates for B2B HR and Talent Acquisition audiences in the UK & Ireland.

Tone:
- calm, professional, human
- no hype, no buzzwords, no exclamation marks
- no emojis
- plain text

Rules:
- do not include links
- do not invent metrics or claims
- include placeholders exactly: {first_name}, {job_title}, {company}
Return exactly:
Subject: <subject>
Body:
<body>
""".strip()


def generate_template_with_ai(campaign: str) -> Optional[Dict[str, str]]:
    if OpenAI is None or not OPENAI_API_KEY:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    if campaign == "Scheduler":
        user_prompt = """
Write an outbound email TEMPLATE for an interview scheduling tool.
Context:
- Reduces back-and-forth coordination
- No calendar integrations
- Hiring manager shares a screenshot/PDF of availability
- Candidate selects a suitable slot
CTA should be soft.
""".strip()
    else:
        user_prompt = """
Write an outbound email TEMPLATE for an HR operational layer product.
Context:
- Helps reduce manual operational work across hiring, onboarding, reporting
- Sits alongside existing HR systems (not replacement)
CTA should be soft.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = resp.choices[0].message.content.strip()
    subject = ""
    body = ""

    # Simple parsing
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
        if line.lower().startswith("body:"):
            body = "\n".join(lines[i + 1 :]).strip()
            break

    if not subject or not body:
        return None
    return {"subject": subject, "body": body}


# -----------------------------
# SMTP
# -----------------------------
def send_email(to_email: str, subject: str, body: str, from_name: str) -> None:
    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError("SMTP_USER/SMTP_PASSWORD not configured. Set in Streamlit secrets or environment variables.")

    msg = MIMEMultipart()
    msg["From"] = f"{from_name} <{SMTP_USER}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


# -----------------------------
# Scheduling helpers
# -----------------------------
def compute_send_time(send_window: str, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)

    def at_today(h: int) -> datetime:
        return now.replace(hour=h, minute=0, second=0, microsecond=0)

    if send_window == "Now":
        return now
    if send_window == "Today 8am":
        return at_today(8)
    if send_window == "Today 11am":
        return at_today(11)
    if send_window == "Today 2pm":
        return at_today(14)
    if send_window == "Tomorrow 8am":
        t = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        return t

    return now


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PowerDash AI SDR", layout="centered")
st.title("PowerDash AI SDR – Outbound Email Sender")

with st.expander("Setup notes (recommended)", expanded=False):
    st.write(
        "- Keep **Dry run** ON until you've sent a test email to yourself.\n"
        "- Use placeholders in templates: `{first_name}`, `{job_title}`, `{company}`.\n"
        "- For Streamlit Cloud, set secrets under **App → Settings → Secrets**."
    )

# Upload multiple CSVs
uploaded_files = st.file_uploader(
    "Upload Apollo CSVs (you can select multiple files)",
    type=["csv"],
    accept_multiple_files=True,
)

# Global controls
colA, colB = st.columns(2)
with colA:
    campaign = st.radio("Campaign", ["Scheduler", "Suite"], horizontal=True)
with colB:
    tz_name = st.selectbox("Timezone", [APP_TZ_DEFAULT, "UTC"], index=0)

templates = load_templates()

# Template editor
st.subheader(f"{campaign} template")
tokens_help = "Available tokens: {first_name}, {job_title}, {company}"

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    if st.button("Save template"):
        templates[campaign]["subject"] = st.session_state.get("tmpl_subject", templates[campaign]["subject"])
        templates[campaign]["body"] = st.session_state.get("tmpl_body", templates[campaign]["body"])
        save_templates(templates)
        st.success("Template saved.")
with btn_col2:
    if st.button("Reset to default"):
        templates[campaign]["subject"] = DEFAULT_TEMPLATES[campaign]["subject"]
        templates[campaign]["body"] = DEFAULT_TEMPLATES[campaign]["body"]
        save_templates(templates)
        st.success("Reset to default and saved.")
with btn_col3:
    if st.button("Generate suggested template with AI"):
        suggestion = generate_template_with_ai(campaign)
        if suggestion is None:
            st.warning("AI template generation unavailable. Check OPENAI_API_KEY and openai dependency.")
        else:
            templates[campaign]["subject"] = suggestion["subject"]
            templates[campaign]["body"] = suggestion["body"]
            save_templates(templates)
            st.success("Suggested template generated and saved. You can edit it below.")

subject_template = st.text_input(
    "Subject",
    value=templates[campaign]["subject"],
    help=tokens_help,
    key="tmpl_subject",
)
body_template = st.text_area(
    "Body",
    value=templates[campaign]["body"],
    height=220,
    help=tokens_help,
    key="tmpl_body",
)

st.divider()

# Sending controls
dry_run = st.checkbox("Dry run (do not send emails)", value=True)
send_window = st.selectbox("Send window", ["Now", "Today 8am", "Today 11am", "Today 2pm", "Tomorrow 8am"])
max_emails = st.number_input("Max emails this run", min_value=1, max_value=200, value=20, step=1)
send_delay = st.number_input("Delay between emails (seconds)", min_value=10, max_value=600, value=DEFAULT_SEND_DELAY_SECONDS, step=10)
from_name = st.text_input("From name", value=FROM_NAME_DEFAULT)

test_to = st.text_input("Optional: test recipient email (send all to this address)", value="")

# Load / normalize leads
if not uploaded_files:
    st.info("Upload one or more Apollo CSVs to continue.")
    st.stop()

try:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    raw = pd.concat(dfs, ignore_index=True)
    leads = normalize_leads(raw)
except Exception as e:
    st.error(f"Could not read/normalize CSV(s): {e}")
    st.stop()

st.subheader("Preview leads")
st.caption(f"Loaded {len(leads)} leads from {len(uploaded_files)} file(s).")

# Lead selection UI
work = leads.copy()
work["send"] = True

# Show a compact editor (first N only)
preview_n = min(int(max_emails), len(work))
st.write("Select which rows to include in this run:")
edited = st.data_editor(
    work.head(preview_n),
    use_container_width=True,
    hide_index=True,
    column_config={
        "send": st.column_config.CheckboxColumn("Send", help="Uncheck to skip this person"),
        "first_name": st.column_config.TextColumn("First name"),
        "job_title": st.column_config.TextColumn("Title"),
        "company": st.column_config.TextColumn("Company"),
        "email": st.column_config.TextColumn("Email"),
    },
    disabled=["first_name", "last_name", "job_title", "company", "email"],
)

selected = edited[edited["send"] == True].copy()
selected.reset_index(drop=True, inplace=True)

st.caption(f"Selected {len(selected)} lead(s) for this run.")

# Preview generated (templated) emails
with st.expander("Preview emails (templated)", expanded=False):
    if len(selected) == 0:
        st.write("No leads selected.")
    else:
        for i, row in selected.iterrows():
            to_email = test_to.strip().lower() if test_to.strip() else row["email"]
            subj = subject_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])
            body = body_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])
            st.markdown(f"**To:** {to_email}")
            st.markdown(f"**Subject:** {subj}")
            st.text(body)
            st.divider()

# Send action
st.subheader("Send")
confirm = st.checkbox("I understand this will send email (unless Dry run is on).", value=False)

send_btn = st.button("Send selected")

if send_btn:
    if not confirm:
        st.warning("Please tick the confirmation checkbox before sending.")
        st.stop()

    target_time = compute_send_time(send_window, tz_name)
    now_tz = datetime.now(ZoneInfo(tz_name))
    wait_seconds = max(0, (target_time - now_tz).total_seconds())

    # Long waits on Streamlit Cloud are fragile; warn user.
    if wait_seconds > 600:
        st.warning(
            f"This run is scheduled for {target_time.strftime('%Y-%m-%d %H:%M %Z')}. "
            f"That’s {int(wait_seconds//60)} minutes from now. "
            "Streamlit Cloud may not reliably keep long sleeps alive unless the browser stays open."
        )
        proceed_long_wait = st.checkbox("Proceed anyway (keep this tab open)", value=False)
        if not proceed_long_wait:
            st.stop()

    if wait_seconds > 0:
        st.info(f"Waiting until {target_time.strftime('%Y-%m-%d %H:%M %Z')} to send…")
        # Simple countdown (best-effort)
        countdown = st.empty()
        remaining = int(wait_seconds)
        while remaining > 0:
            countdown.write(f"Sending starts in {remaining} seconds…")
            time.sleep(1)
            remaining -= 1
        countdown.empty()

    # Process send
    log_rows: List[Dict[str, Any]] = []
    sent_count = 0

    progress = st.progress(0)
    status = st.empty()

    total = len(selected)
    for i, row in selected.iterrows():
        to_email = test_to.strip().lower() if test_to.strip() else row["email"]

        subj = subject_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])
        body = body_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])

        ts = datetime.now(ZoneInfo(tz_name)).isoformat(timespec="seconds")

        try:
            if not dry_run:
                send_email(to_email, subj, body, from_name=from_name)
                time.sleep(int(send_delay))
            result = "SENT" if not dry_run else "DRY_RUN"
            error = ""
            sent_count += 1
        except Exception as e:
            result = "ERROR"
            error = str(e)

        log_rows.append(
            {
                "timestamp": ts,
                "campaign": campaign,
                "to_email": to_email,
                "first_name": row["first_name"],
                "job_title": row["job_title"],
                "company": row["company"],
                "subject": subj,
                "body": body,
                "result": result,
                "error": error,
            }
        )

        progress.progress(min(1.0, (i + 1) / max(1, total)))
        status.write(f"Processed {i+1}/{total}… ({result})")

    status.empty()
    progress.empty()

    st.success(f"Processed {sent_count} email(s). Result: {'DRY RUN' if dry_run else 'LIVE SEND'}")

    # Show / download logs
    log_df = pd.DataFrame(log_rows)
    st.subheader("Run log")
    st.dataframe(log_df[["timestamp", "campaign", "to_email", "company", "job_title", "result", "error"]], use_container_width=True)

    csv_bytes = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download run log (CSV)",
        data=csv_bytes,
        file_name=f"powerdash_ai_sdr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
