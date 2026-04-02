import os
import json
import time
import smtplib
import socket
import hashlib
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List, Iterator

import pandas as pd
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


APP_TZ_DEFAULT = "Europe/London"
TEMPLATES_FILE = "templates.json"
STATE_FILE = "send_state.json"
RUN_LOG_DIR = Path("run_logs")
RUN_LOG_DIR.mkdir(exist_ok=True)

FROM_NAME_DEFAULT = "Martin McDonald"
DEFAULT_SEND_DELAY_SECONDS = 120
DEFAULT_SMTP_TIMEOUT_SECONDS = 30
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_WAIT_SECONDS = 8


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


TRANSIENT_SMTP_EXCEPTIONS = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    smtplib.SMTPHeloError,
    smtplib.SMTPDataError,
    smtplib.SMTPRecipientsRefused,
    TimeoutError,
    socket.timeout,
)


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


# -----------------------------
# Persistence helpers
# -----------------------------
def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# Template helpers
# -----------------------------
def load_templates() -> Dict[str, Dict[str, str]]:
    templates = json.loads(json.dumps(DEFAULT_TEMPLATES))
    saved = load_json_file(TEMPLATES_FILE, {})
    for k in templates.keys():
        if isinstance(saved, dict) and k in saved and isinstance(saved[k], dict):
            templates[k]["subject"] = str(saved[k].get("subject", templates[k]["subject"]))
            templates[k]["body"] = str(saved[k].get("body", templates[k]["body"]))
    return templates


def save_templates(templates: Dict[str, Dict[str, str]]) -> None:
    save_json_file(TEMPLATES_FILE, templates)


# -----------------------------
# Lead helpers
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

    out["first_name"] = out["first_name"].str.strip()
    out["job_title"] = out["job_title"].str.strip()
    out["company"] = out["company"].str.strip()
    out["email"] = out["email"].str.strip().str.lower()
    out = out[out["email"].astype(bool)].copy()
    out = out.drop_duplicates(subset=["email"], keep="first").reset_index(drop=True)
    return out


# -----------------------------
# Template suggestion
# -----------------------------
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
# SMTP helpers
# -----------------------------
def build_message(to_email: str, subject: str, body: str, from_name: str) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg["From"] = f"{from_name} <{SMTP_USER}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


@contextmanager
def smtp_session() -> Iterator[smtplib.SMTP]:
    if not SMTP_USER or not SMTP_PASSWORD:
        raise RuntimeError("SMTP_USER/SMTP_PASSWORD not configured. Set in Streamlit secrets or environment variables.")

    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=DEFAULT_SMTP_TIMEOUT_SECONDS)
    try:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_USER, SMTP_PASSWORD)
        yield server
    finally:
        try:
            server.quit()
        except Exception:
            try:
                server.close()
            except Exception:
                pass


def send_with_retry(server: smtplib.SMTP, to_email: str, subject: str, body: str, from_name: str, retry_count: int) -> None:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retry_count + 1):
        try:
            msg = build_message(to_email, subject, body, from_name)
            server.send_message(msg)
            return
        except TRANSIENT_SMTP_EXCEPTIONS as exc:
            last_exc = exc
            if attempt >= retry_count:
                break
            time.sleep(DEFAULT_RETRY_WAIT_SECONDS * attempt)
            server.ehlo()
        except Exception:
            raise
    if last_exc is not None:
        raise last_exc


# -----------------------------
# Scheduling / run state
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
        return (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    return now


def make_run_id(records: List[Dict[str, Any]], campaign: str, dry_run: bool, test_to: str) -> str:
    fingerprint_source = json.dumps(
        {
            "campaign": campaign,
            "dry_run": dry_run,
            "test_to": test_to,
            "emails": [r.get("email", "") for r in records],
        },
        sort_keys=True,
    )
    return hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12]


def run_log_path(run_id: str) -> Path:
    return RUN_LOG_DIR / f"powerdash_ai_sdr_log_{run_id}.csv"


def load_run_state() -> Dict[str, Any]:
    return load_json_file(STATE_FILE, {})


def save_run_state(state: Dict[str, Any]) -> None:
    save_json_file(STATE_FILE, state)


def clear_run_state() -> None:
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def append_log_row(log_path: Path, row: Dict[str, Any]) -> None:
    row_df = pd.DataFrame([row])
    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, index=False)


def initialize_run_state(
    selected_df: pd.DataFrame,
    campaign: str,
    subject_template: str,
    body_template: str,
    dry_run: bool,
    send_delay: int,
    from_name: str,
    test_to: str,
    tz_name: str,
) -> Dict[str, Any]:
    records = selected_df[["first_name", "last_name", "job_title", "company", "email"]].to_dict(orient="records")
    run_id = make_run_id(records, campaign, dry_run, test_to.strip().lower())
    log_path = run_log_path(run_id)
    state = {
        "run_id": run_id,
        "status": "running",
        "created_at": datetime.now(ZoneInfo(tz_name)).isoformat(timespec="seconds"),
        "campaign": campaign,
        "subject_template": subject_template,
        "body_template": body_template,
        "dry_run": dry_run,
        "send_delay": int(send_delay),
        "from_name": from_name,
        "test_to": test_to.strip().lower(),
        "tz_name": tz_name,
        "records": records,
        "completed_indexes": [],
        "sent_indexes": [],
        "error_indexes": [],
        "log_path": str(log_path),
    }
    save_run_state(state)
    return state


def mark_run_complete(state: Dict[str, Any]) -> None:
    state["status"] = "completed"
    state["completed_at"] = datetime.now(ZoneInfo(state["tz_name"])).isoformat(timespec="seconds")
    save_run_state(state)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PowerDash AI SDR", layout="centered")
st.title("PowerDash AI SDR – Outbound Email Sender")

with st.expander("Setup notes (recommended)", expanded=False):
    st.write(
        "- Keep **Dry run** ON until you've sent a test email to yourself.\n"
        "- Use placeholders in templates: `{first_name}`, `{job_title}`, `{company}`.\n"
        "- For Streamlit Cloud, set secrets under **App → Settings → Secrets**.\n"
        "- This version reuses one SMTP connection, writes progress after every email, and can resume an interrupted run."
    )

state = load_run_state()
if state and state.get("status") == "running":
    completed = len(state.get("completed_indexes", []))
    total_saved = len(state.get("records", []))
    sent_saved = len(state.get("sent_indexes", []))
    st.warning(
        f"Unfinished run detected: {completed}/{total_saved} processed, {sent_saved} sent. "
        f"Run ID: {state.get('run_id', 'unknown')}"
    )

uploaded_files = st.file_uploader(
    "Upload Apollo CSVs (you can select multiple files)",
    type=["csv"],
    accept_multiple_files=True,
)

colA, colB = st.columns(2)
with colA:
    campaign = st.radio("Campaign", ["Scheduler", "Suite"], horizontal=True)
with colB:
    tz_name = st.selectbox("Timezone", [APP_TZ_DEFAULT, "UTC"], index=0)

templates = load_templates()

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

subject_template = st.text_input("Subject", value=templates[campaign]["subject"], help=tokens_help, key="tmpl_subject")
body_template = st.text_area("Body", value=templates[campaign]["body"], height=220, help=tokens_help, key="tmpl_body")

st.divider()

dry_run = st.checkbox("Dry run (do not send emails)", value=True)
send_window = st.selectbox("Send window", ["Now", "Today 8am", "Today 11am", "Today 2pm", "Tomorrow 8am"])
max_emails = st.number_input("Max emails this run", min_value=1, max_value=200, value=20, step=1)
send_delay = st.number_input("Delay between emails (seconds)", min_value=10, max_value=600, value=DEFAULT_SEND_DELAY_SECONDS, step=10)
from_name = st.text_input("From name", value=FROM_NAME_DEFAULT)
test_to = st.text_input("Optional: test recipient email (send all to this address)", value="")
retry_count = st.number_input("Retry attempts per email", min_value=1, max_value=5, value=DEFAULT_RETRY_COUNT, step=1)

selected = pd.DataFrame()
if not uploaded_files:
    st.info("Upload one or more Apollo CSVs to continue.")
else:
    try:
        dfs = [pd.read_csv(f) for f in uploaded_files]
        raw = pd.concat(dfs, ignore_index=True)
        leads = normalize_leads(raw)
    except Exception as e:
        st.error(f"Could not read/normalize CSV(s): {e}")
        st.stop()

    st.subheader("Preview leads")
    st.caption(f"Loaded {len(leads)} leads from {len(uploaded_files)} file(s).")

    work = leads.copy()
    work["send"] = True
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

    selected = edited[edited["send"] == True].copy().reset_index(drop=True)
    st.caption(f"Selected {len(selected)} lead(s) for this run.")

    with st.expander("Preview emails (templated)", expanded=False):
        if len(selected) == 0:
            st.write("No leads selected.")
        else:
            for _, row in selected.iterrows():
                to_email = test_to.strip().lower() if test_to.strip() else row["email"]
                subj = subject_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])
                body = body_template.format(first_name=row["first_name"], job_title=row["job_title"], company=row["company"])
                st.markdown(f"**To:** {to_email}")
                st.markdown(f"**Subject:** {subj}")
                st.text(body)
                st.divider()

st.subheader("Send")
confirm = st.checkbox("I understand this will send email (unless Dry run is on).", value=False)
resume_btn = st.button("Resume unfinished run") if state and state.get("status") == "running" else False
send_btn = st.button("Send selected")

if resume_btn and not state:
    st.warning("No unfinished run found.")
    st.stop()

if send_btn or resume_btn:
    if not confirm:
        st.warning("Please tick the confirmation checkbox before sending.")
        st.stop()

    if resume_btn:
        run_state = state
        if not run_state:
            st.error("No unfinished run found.")
            st.stop()
        target_time = datetime.now(ZoneInfo(run_state["tz_name"]))
        selected_records = run_state["records"]
        dry_run_mode = bool(run_state["dry_run"])
        delay_seconds = int(run_state["send_delay"])
        active_from_name = run_state["from_name"]
        active_retry_count = int(retry_count)
    else:
        if selected.empty:
            st.warning("No leads selected.")
            st.stop()
        run_state = initialize_run_state(
            selected_df=selected,
            campaign=campaign,
            subject_template=subject_template,
            body_template=body_template,
            dry_run=dry_run,
            send_delay=int(send_delay),
            from_name=from_name,
            test_to=test_to,
            tz_name=tz_name,
        )
        target_time = compute_send_time(send_window, tz_name)
        selected_records = run_state["records"]
        dry_run_mode = bool(dry_run)
        delay_seconds = int(send_delay)
        active_from_name = from_name
        active_retry_count = int(retry_count)

    now_tz = datetime.now(ZoneInfo(run_state["tz_name"]))
    wait_seconds = max(0, (target_time - now_tz).total_seconds())

    if wait_seconds > 600:
        st.warning(
            f"This run is scheduled for {target_time.strftime('%Y-%m-%d %H:%M %Z')}. "
            f"That’s {int(wait_seconds // 60)} minutes from now. Long in-app waits are still fragile."
        )
        proceed_long_wait = st.checkbox("Proceed anyway (keep this tab open)", value=False, key="proceed_long_wait")
        if not proceed_long_wait:
            st.stop()

    if wait_seconds > 0:
        st.info(f"Waiting until {target_time.strftime('%Y-%m-%d %H:%M %Z')} to send…")
        countdown = st.empty()
        remaining = int(wait_seconds)
        while remaining > 0:
            countdown.write(f"Sending starts in {remaining} seconds…")
            time.sleep(1)
            remaining -= 1
        countdown.empty()

    progress = st.progress(0)
    status = st.empty()
    completed_indexes = set(run_state.get("completed_indexes", []))
    total = len(selected_records)
    log_path = Path(run_state["log_path"])

    try:
        if dry_run_mode:
            smtp_ctx = None
        else:
            smtp_ctx = smtp_session()

        if smtp_ctx is None:
            server_cm = None
            server_obj = None
        else:
            server_cm = smtp_ctx
            server_obj = server_cm.__enter__()

        try:
            for i, row in enumerate(selected_records):
                if i in completed_indexes:
                    progress.progress(min(1.0, (i + 1) / max(1, total)))
                    continue

                to_email = run_state["test_to"] if run_state.get("test_to") else row["email"]
                subj = run_state["subject_template"].format(
                    first_name=row["first_name"], job_title=row["job_title"], company=row["company"]
                )
                body = run_state["body_template"].format(
                    first_name=row["first_name"], job_title=row["job_title"], company=row["company"]
                )
                ts = datetime.now(ZoneInfo(run_state["tz_name"])).isoformat(timespec="seconds")

                try:
                    if dry_run_mode:
                        result = "DRY_RUN"
                        error = ""
                    else:
                        assert server_obj is not None
                        send_with_retry(server_obj, to_email, subj, body, active_from_name, active_retry_count)
                        result = "SENT"
                        error = ""
                    run_state.setdefault("sent_indexes", [])
                    if result == "SENT" and i not in run_state["sent_indexes"]:
                        run_state["sent_indexes"].append(i)
                except Exception as e:
                    result = "ERROR"
                    error = str(e)
                    run_state.setdefault("error_indexes", [])
                    if i not in run_state["error_indexes"]:
                        run_state["error_indexes"].append(i)

                log_row = {
                    "timestamp": ts,
                    "run_id": run_state["run_id"],
                    "campaign": run_state["campaign"],
                    "to_email": to_email,
                    "first_name": row["first_name"],
                    "job_title": row["job_title"],
                    "company": row["company"],
                    "subject": subj,
                    "body": body,
                    "result": result,
                    "error": error,
                }
                append_log_row(log_path, log_row)

                run_state.setdefault("completed_indexes", [])
                if i not in run_state["completed_indexes"]:
                    run_state["completed_indexes"].append(i)
                save_run_state(run_state)

                progress.progress(min(1.0, (i + 1) / max(1, total)))
                status.write(f"Processed {i + 1}/{total}… ({result})")

                if i < total - 1 and delay_seconds > 0:
                    time.sleep(delay_seconds)
        finally:
            if server_cm is not None:
                server_cm.__exit__(None, None, None)

    except Exception as e:
        st.error(f"Run stopped unexpectedly: {e}")
        st.info("Progress was saved after each processed row. You can click 'Resume unfinished run'.")
        st.stop()

    status.empty()
    progress.empty()
    mark_run_complete(run_state)

    log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    sent_count = int((log_df["result"] == "SENT").sum()) if not log_df.empty else 0
    dry_run_count = int((log_df["result"] == "DRY_RUN").sum()) if not log_df.empty else 0
    error_count = int((log_df["result"] == "ERROR").sum()) if not log_df.empty else 0

    st.success(
        f"Run complete. Sent: {sent_count} | Dry run: {dry_run_count} | Errors: {error_count}"
    )
    st.subheader("Run log")
    if log_df.empty:
        st.write("No log rows were written.")
    else:
        st.dataframe(
            log_df[["timestamp", "campaign", "to_email", "company", "job_title", "result", "error"]],
            use_container_width=True,
        )
        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download run log (CSV)",
            data=csv_bytes,
            file_name=log_path.name,
            mime="text/csv",
        )
