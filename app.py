import streamlit as st
import pandas as pd
import smtplib
import time
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# ENV
# -------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

FROM_NAME = "Martin McDonald"

# -------------------------------------------------
# PROMPTS (unchanged)
# -------------------------------------------------
SYSTEM_PROMPT = """(same as before)"""

def scheduler_prompt(first_name, job_title, company):
    return f"""(same improved Scheduler prompt)"""

def suite_prompt(first_name, job_title, company):
    return f"""(same Suite prompt)"""

# -------------------------------------------------
# SMTP
# -------------------------------------------------
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = f"{FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("PowerDash AI SDR – Outbound Email Sender")

uploaded_file = st.file_uploader("Upload Apollo CSV", type=["csv"])

campaign_mode = st.radio(
    "Campaign mode",
    ["Auto (by job title)", "Scheduler", "Suite"]
)

send_window = st.selectbox(
    "Send window",
    ["Now", "Today 8am", "Today 11am", "Today 2pm", "Tomorrow 8am"]
)

dry_run = st.checkbox("Dry run (do not send emails)", value=True)
max_emails = st.number_input("Max emails this run", 1, 50, 10)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview leads")
st.dataframe(df.head(10))

if st.button("Generate Emails"):
    generated = []

    for _, row in df.head(max_emails).iterrows():
        first = row["First Name"]
        title = row["Title"]
        company = row["Company Name"]
        email = row["Email"]

        if campaign_mode == "Scheduler":
            prompt = scheduler_prompt(first, title, company)
        elif campaign_mode == "Suite":
            prompt = suite_prompt(first, title, company)
        else:
            prompt = (
                scheduler_prompt(first, title, company)
                if "talent" in title.lower()
                else suite_prompt(first, title, company)
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content
        lines = text.splitlines()
        subject = lines[0].replace("Subject:", "").strip()
        body = "\n".join(lines[1:]).strip()

        generated.append({
            "email": email,
            "subject": subject,
            "body": body
        })

    st.session_state["emails"] = generated

# -------------------------------------------------
# EDIT & SEND
# -------------------------------------------------
if "emails" in st.session_state:
    st.subheader("Review & edit emails")

    for i, e in enumerate(st.session_state["emails"]):
        st.markdown(f"### {e['email']}")
        e["subject"] = st.text_input("Subject", e["subject"], key=f"s{i}")
        e["body"] = st.text_area("Body", e["body"], height=200, key=f"b{i}")

    if st.button("Send Emails"):
        if send_window != "Now":
            target = {
                "Today 8am": datetime.now().replace(hour=8, minute=0),
                "Today 11am": datetime.now().replace(hour=11, minute=0),
                "Today 2pm": datetime.now().replace(hour=14, minute=0),
                "Tomorrow 8am": (datetime.now() + timedelta(days=1)).replace(hour=8, minute=0),
            }[send_window]

            wait_seconds = max(0, (target - datetime.now()).total_seconds())
            st.info(f"Waiting until {target.strftime('%H:%M')} to send…")
            time.sleep(wait_seconds)

        for e in st.session_state["emails"]:
            if not dry_run:
                send_email(e["email"], e["subject"], e["body"])
                time.sleep(120)

        st.success("Emails processed.")
