import pandas as pd
import smtplib
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CSV_FILE = "scheduler_pilot_25_batch1.csv"
SEND_DELAY_SECONDS = 120  # 2 minutes between emails
FROM_NAME = "Martin McDonald"

# -------------------------------------------------
# SYSTEM PROMPT (shared)
# -------------------------------------------------
SYSTEM_PROMPT = """
You are writing outbound sales emails on behalf of Martin McDonald, founder of PowerDash HR.

Tone rules:
- Professional, calm, human, and founder-led
- No hype, no buzzwords, no exclamation marks
- No emojis
- No marketing language

Style rules:
- Plain text email only
- Maximum 90 words
- Short paragraphs (1–2 lines)
- No bullet points
- No links
- One clear, soft call to action

Content rules:
- Do not invent facts about the recipient or their company
- Do not reference news, funding, or growth unless explicitly provided
- Do not assume pain — ask lightly instead
- Never mention AI explicitly

Signature:
Martin
Martin McDonald
PowerDash HR
"""

# -------------------------------------------------
# PROMPT BUILDERS
# -------------------------------------------------
def scheduler_prompt(first_name, job_title, company):
    return f"""
Write a short cold email to {first_name}, who is a {job_title} at {company}.

Product context:
- Interview scheduling tool for TA teams
- Removes back-and-forth coordination
- No calendar integrations required
- Hiring managers share a screenshot or PDF of availability
- Candidates select a slot themselves

Email goals:
- Acknowledge their role briefly
- Describe the problem in neutral terms
- Mention the product as a simple, practical option
- Ask if it is worth a quick look

Return:
- A subject line (max 6 words)
- The email body
"""

def suite_prompt(first_name, job_title, company):
    return f"""
Write a short cold email to {first_name}, who is a {job_title} at {company}.

Product context:
- AI-enabled operational layer for HR teams
- Reduces manual effort across hiring, onboarding, and reporting
- Works alongside existing HR systems
- Focused on execution, not replacement

Email goals:
- Acknowledge senior responsibility
- Reference operational friction carefully
- Position PowerDash as lightweight
- Suggest an exploratory conversation

Return:
- A subject line (max 6 words)
- The email body
"""

# -------------------------------------------------
# OPENAI CALL
# -------------------------------------------------
def generate_email(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------
# SMTP SEND
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
# MAIN
# -------------------------------------------------
def main():
    df = pd.read_csv(CSV_FILE)

    # basic routing
    def route_campaign(title):
        title = str(title).lower()
        if "talent" in title or "recruit" in title:
            return "scheduler"
        return "suite"

    for index, row in df.iterrows():
        first_name = row["First Name"]
        job_title = row["Title"]
        company = row["Company Name"]
        email = row["Email"]

        campaign = route_campaign(job_title)

        if campaign == "scheduler":
            prompt = scheduler_prompt(first_name, job_title, company)
        else:
            prompt = suite_prompt(first_name, job_title, company)

        print(f"Generating email for {first_name} ({campaign})")

        output = generate_email(prompt)

        # VERY simple parsing (subject first line)
        lines = output.splitlines()
        subject = lines[0].replace("Subject:", "").strip()
        body = "\n".join(lines[1:]).strip()

        print("Subject:", subject)
        print(body)
        print("Sending to:", email)
        print("-----")

        send_email(email, subject, body)

        time.sleep(SEND_DELAY_SECONDS)

    print("Done.")

if __name__ == "__main__":
    main()
