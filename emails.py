"""
Email dataset for the Email Triage OpenEnv environment.
20 diverse real-world emails with ground truth labels for all 3 tasks.

Ground truth fields:
  label    : spam | inbox | urgent | archive | delete
  priority : high | medium | low
  category : spam | work | personal | newsletter | notification | social
"""

EMAILS = [
    # ─────────────────────────────────────────────────────────────
    # SPAM / PHISHING (4 emails)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_001",
        "subject": "CONGRATULATIONS! You've Won $1,000,000!!!",
        "sender": "noreply@lottery-winners.tk",
        "body": (
            "Dear Lucky Winner! You have been selected as the winner of our "
            "international lottery! To claim your prize of ONE MILLION DOLLARS, "
            "please send your bank details and a processing fee of $500 to "
            "claim@lottery-winners.tk. This offer expires in 24 hours! "
            "Act NOW before your prize is forfeited!"
        ),
        "timestamp": "2024-01-15 09:23:00",
        "ground_truth": {"label": "spam", "priority": "low", "category": "spam"},
    },
    {
        "id": "email_002",
        "subject": "Make $5,000/week working from home — Limited spots!",
        "sender": "opportunities@quick-cash-now.biz",
        "body": (
            "Are you tired of your 9-5? Our proven system lets you earn $5,000 "
            "EVERY WEEK from the comfort of your home! No experience needed! "
            "No investment required! 100% guaranteed! Click here to claim your "
            "spot before it's gone. Only 3 spots left in your area! Act NOW!"
        ),
        "timestamp": "2024-01-15 11:45:00",
        "ground_truth": {"label": "spam", "priority": "low", "category": "spam"},
    },
    {
        "id": "email_003",
        "subject": "Your PayPal account has been suspended — Verify immediately",
        "sender": "security@paypa1-account-verify.com",
        "body": (
            "Dear PayPal Customer, We detected unusual activity on your account. "
            "Your account has been temporarily suspended for your protection. "
            "Click here to verify your identity: http://paypa1-verify.com/login. "
            "You must act within 24 hours or your account will be permanently "
            "closed. Please enter your username, password, and SSN to restore access."
        ),
        "timestamp": "2024-01-15 14:12:00",
        "ground_truth": {"label": "spam", "priority": "low", "category": "spam"},
    },
    {
        "id": "email_004",
        "subject": "RE: Business Partnership Proposal — $4.7M Transfer",
        "sender": "dr.james.okoro@businesspartner.ng",
        "body": (
            "Dear Friend, I am Dr. James Okoro, a senior official at the Central "
            "Bank of Nigeria. I have a business proposal that will be of great "
            "benefit to both of us. I need a foreign partner to help me transfer "
            "$4.7 million USD. You will receive 30% of the total amount. "
            "Please reply with your full name and bank details to proceed. "
            "This is 100% legal and risk-free."
        ),
        "timestamp": "2024-01-15 16:30:00",
        "ground_truth": {"label": "spam", "priority": "low", "category": "spam"},
    },

    # ─────────────────────────────────────────────────────────────
    # URGENT WORK EMAILS (5 emails)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_005",
        "subject": "CRITICAL [P0]: Production database is DOWN — All engineers needed",
        "sender": "alerts@ops.company.com",
        "body": (
            "CRITICAL ALERT [P0 INCIDENT]: Production database cluster "
            "prod-db-primary has been unreachable for 8 minutes. "
            "Error rate: 100%. Users affected: 127,000. "
            "Revenue impact: ~$45,000/minute. "
            "Incident #4521 has been created in PagerDuty. "
            "On-call team: join the war room immediately → https://meet.company.com/incident-4521. "
            "All engineers please stand by for instructions."
        ),
        "timestamp": "2024-01-15 03:47:00",
        "ground_truth": {"label": "urgent", "priority": "high", "category": "work"},
    },
    {
        "id": "email_006",
        "subject": "Board meeting moved to TODAY 2PM — Leadership attendance mandatory",
        "sender": "ceo@company.com",
        "body": (
            "Team — Due to the investor situation that developed overnight, "
            "I need ALL leadership in the board room at 2PM today. "
            "This is mandatory — please cancel all other commitments. "
            "We are discussing Q4 results and the Series B bridge situation. "
            "If you cannot attend in person, dial in: +1-555-0100, code 8821. "
            "This takes absolute priority over everything else today. — Sarah (CEO)"
        ),
        "timestamp": "2024-01-15 08:15:00",
        "ground_truth": {"label": "urgent", "priority": "high", "category": "work"},
    },
    {
        "id": "email_007",
        "subject": "URGENT: AcmeCorp threatening to cancel $2M contract — Need CTO today",
        "sender": "john.smith@sales.company.com",
        "body": (
            "I just got off the phone with AcmeCorp. They are extremely unhappy "
            "about the last 3 service outages. Their CEO is threatening to cancel "
            "their $2M annual contract unless they get a call with our CTO by 5PM EST TODAY. "
            "AcmeCorp is our single largest client — we cannot lose them. "
            "I have tried calling but need help escalating. "
            "Please connect me with CTO's office immediately. This CANNOT wait."
        ),
        "timestamp": "2024-01-15 10:22:00",
        "ground_truth": {"label": "urgent", "priority": "high", "category": "work"},
    },
    {
        "id": "email_008",
        "subject": "SECURITY INCIDENT: Employee credentials exposed — Exec briefing required",
        "sender": "security@company.com",
        "body": (
            "SECURITY INCIDENT REPORT [CONFIDENTIAL]: Our security team has detected "
            "unauthorized access to our HR system. Approximately 240 employee records "
            "may have been exfiltrated, including names, addresses, and SSNs. "
            "We have isolated the system and engaged our IR firm. "
            "Legal and executive team must be briefed immediately. "
            "Do NOT discuss on Slack or email. Emergency meeting: Conference Room A, 15 minutes."
        ),
        "timestamp": "2024-01-15 13:05:00",
        "ground_truth": {"label": "urgent", "priority": "high", "category": "work"},
    },
    {
        "id": "email_009",
        "subject": "Regulatory deadline moved up — Full report needed by THIS Friday",
        "sender": "manager@company.com",
        "body": (
            "Team — Just confirmed with the client: the regulatory submission deadline "
            "has been pulled forward. We now need the COMPLETE report by this Friday "
            "EOD instead of next week. This is non-negotiable — the client faces a "
            "$500K fine if we miss it. "
            "Please assess what is achievable and confirm your availability for "
            "extra hours this week. I need your plan by end of today. "
            "This is our #1 priority right now."
        ),
        "timestamp": "2024-01-15 09:00:00",
        "ground_truth": {"label": "urgent", "priority": "high", "category": "work"},
    },

    # ─────────────────────────────────────────────────────────────
    # NORMAL WORK EMAILS (4 emails)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_010",
        "subject": "Notes from yesterday's Q2 product planning session",
        "sender": "alice@company.com",
        "body": (
            "Hi team — Attached are the meeting notes from yesterday's Q2 planning. "
            "Key decisions: 1) Q2 roadmap approved with 3 features. "
            "2) Design reviews every Thursday 3PM. "
            "3) Beta launch target: March 15th. "
            "Action items have been created in Jira — please review yours. "
            "Let me know if I missed anything. Thanks, Alice"
        ),
        "timestamp": "2024-01-15 08:30:00",
        "ground_truth": {"label": "inbox", "priority": "medium", "category": "work"},
    },
    {
        "id": "email_011",
        "subject": "Quick question about the payment gateway auth pattern",
        "sender": "dev.colleague@company.com",
        "body": (
            "Hey — Hope you're well! When you built the payment gateway integration "
            "last quarter, did you use OAuth 2.0 or API keys for authentication? "
            "I'm starting a similar project and want to follow the established pattern "
            "so we stay consistent. No rush on this at all — whenever you get a chance. "
            "Thanks!"
        ),
        "timestamp": "2024-01-15 14:00:00",
        "ground_truth": {"label": "inbox", "priority": "medium", "category": "work"},
    },
    {
        "id": "email_012",
        "subject": "PR #847 ready for review — auth refactor (sprint ends Thursday)",
        "sender": "bob@company.com",
        "body": (
            "Hi — Could you review PR #847 when you get a chance? "
            "It's the user authentication refactor we scoped in planning. "
            "Sprint ends Thursday so ideally before then, but no hard urgency. "
            "~200 lines of changes with detailed inline comments. "
            "Link: https://github.com/company/repo/pull/847"
        ),
        "timestamp": "2024-01-15 11:00:00",
        "ground_truth": {"label": "inbox", "priority": "medium", "category": "work"},
    },
    {
        "id": "email_013",
        "subject": "Team lunch Friday — RSVP by Wednesday",
        "sender": "hr@company.com",
        "body": (
            "Hi everyone! We're organizing a team lunch this Friday at Pasta Palace "
            "(2 blocks from the office) at 12:30 PM to celebrate Q4 wins! "
            "The company is covering the bill. "
            "Please RSVP by Wednesday so we can confirm the reservation. "
            "Reply YES or NO to this email. Hope to see everyone there! — HR Team"
        ),
        "timestamp": "2024-01-15 10:00:00",
        "ground_truth": {"label": "inbox", "priority": "low", "category": "social"},
    },

    # ─────────────────────────────────────────────────────────────
    # NEWSLETTERS / PROMOTIONAL (3 emails)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_014",
        "subject": "This Week in AI: GPT-5 capabilities, new robotics breakthrough",
        "sender": "newsletter@technews.io",
        "body": (
            "THIS WEEK IN AI | Issue #247\n\n"
            "• GPT-5 capabilities allegedly leaked by insider source\n"
            "• Boston Dynamics unveils warehouse robot achieving 99.1% pick accuracy\n"
            "• EU passes landmark AI liability regulation bill\n"
            "• OpenAI valuation reportedly hits $157B in latest secondary round\n\n"
            "IN DEPTH: How retrieval-augmented generation is reshaping enterprise workflows...\n\n"
            "You're receiving this because you subscribed at technews.io. Unsubscribe here."
        ),
        "timestamp": "2024-01-15 07:00:00",
        "ground_truth": {"label": "archive", "priority": "low", "category": "newsletter"},
    },
    {
        "id": "email_015",
        "subject": "FLASH SALE: 60% off everything — Today only, ends midnight",
        "sender": "deals@fashionstore.com",
        "body": (
            "🔥 FLASH SALE — 24 HOURS ONLY 🔥\n\n"
            "Get 60% off EVERYTHING sitewide — no code needed, discount applied at checkout.\n"
            "Shop our new spring collection, bestselling basics, accessories, and more.\n"
            "Free shipping on orders over $50.\n\n"
            "Shop now → fashionstore.com/sale\n\n"
            "You're receiving this because you opted into promotional emails. "
            "Unsubscribe from promotional emails."
        ),
        "timestamp": "2024-01-15 06:00:00",
        "ground_truth": {"label": "archive", "priority": "low", "category": "newsletter"},
    },
    {
        "id": "email_016",
        "subject": "Weekly Digest — Product updates, blog posts, and community highlights",
        "sender": "digest@productcompany.com",
        "body": (
            "YOUR WEEKLY DIGEST\n\n"
            "NEW THIS WEEK:\n"
            "• v3.2.1 released — 40% faster API response times\n"
            "• New integration: Slack, Notion, and Linear connectors now in beta\n"
            "• Blog: 'How we scaled to 1M users on a $200/month infrastructure'\n\n"
            "COMMUNITY HIGHLIGHTS: Top templates, featured workflows, user spotlights.\n\n"
            "Manage your email preferences | Unsubscribe"
        ),
        "timestamp": "2024-01-15 07:30:00",
        "ground_truth": {"label": "archive", "priority": "low", "category": "newsletter"},
    },

    # ─────────────────────────────────────────────────────────────
    # AUTOMATED NOTIFICATIONS (3 emails)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_017",
        "subject": "PR #847: Bob Chen approved your pull request",
        "sender": "notifications@github.com",
        "body": (
            "Bob Chen approved pull request #847 in company/main-repo.\n\n"
            "PR Title: User authentication refactor\n"
            "Status: APPROVED ✓\n"
            "Comment: 'Great work! Clean implementation. Left 2 minor suggestions '  "
            "'but nothing blocking. LGTM.'\n\n"
            "View PR: https://github.com/company/repo/pull/847\n\n"
            "You're receiving this because you're subscribed to this repository. "
            "Manage notifications | Unsubscribe."
        ),
        "timestamp": "2024-01-15 15:30:00",
        "ground_truth": {"label": "archive", "priority": "low", "category": "notification"},
    },
    {
        "id": "email_018",
        "subject": "ACTION REQUIRED: Annual performance self-review due this Friday",
        "sender": "hr-system@company.com",
        "body": (
            "AUTOMATED REMINDER — ACTION REQUIRED\n\n"
            "Your annual performance self-review is due this Friday, January 19th. "
            "Our records show you have NOT yet completed your self-assessment in the HR portal.\n\n"
            "This is required for all employees. Your manager cannot begin their review "
            "until you complete yours, which may delay your promotion and compensation review.\n\n"
            "Complete now: https://hr.company.com/self-review\n\n"
            "Questions? Contact hr@company.com — Automated HR System"
        ),
        "timestamp": "2024-01-15 09:00:00",
        "ground_truth": {"label": "inbox", "priority": "medium", "category": "notification"},
    },
    {
        "id": "email_019",
        "subject": "Your January bank statement is available",
        "sender": "statements@mybank.com",
        "body": (
            "Your January 2024 statement for account ending in 4521 is now available.\n\n"
            "Account Summary:\n"
            "Opening Balance: $4,521.33\n"
            "Total Credits: $5,250.00\n"
            "Total Debits: $4,879.16 (47 transactions)\n"
            "Closing Balance: $4,892.17\n\n"
            "View your full statement: https://mybank.com/statements\n\n"
            "This is an automated message. Do not reply. "
            "For support, log in to your account or call 1-800-555-BANK."
        ),
        "timestamp": "2024-01-15 08:00:00",
        "ground_truth": {"label": "archive", "priority": "low", "category": "notification"},
    },

    # ─────────────────────────────────────────────────────────────
    # PERSONAL EMAILS (1 email)
    # ─────────────────────────────────────────────────────────────
    {
        "id": "email_020",
        "subject": "Thanksgiving weekend — are you coming this year?",
        "sender": "mom@gmail.com",
        "body": (
            "Hi honey! Just following up on Thanksgiving plans. "
            "Your aunt, uncle, and cousins are all coming this year — it'll be a full house! "
            "We're planning dinner at 3PM. Dad is making his famous apple pie. "
            "Your room is all ready if you want to stay the weekend. "
            "Let me know as soon as you can so I can plan the food shopping. "
            "Love you lots! — Mom\n\n"
            "P.S. Please call your grandmother, she keeps asking about you!"
        ),
        "timestamp": "2024-01-15 12:00:00",
        "ground_truth": {"label": "inbox", "priority": "low", "category": "personal"},
    },
]

# Quick summary for debugging
EMAIL_STATS = {
    "total": len(EMAILS),
    "by_label": {
        "spam": sum(1 for e in EMAILS if e["ground_truth"]["label"] == "spam"),
        "urgent": sum(1 for e in EMAILS if e["ground_truth"]["label"] == "urgent"),
        "inbox": sum(1 for e in EMAILS if e["ground_truth"]["label"] == "inbox"),
        "archive": sum(1 for e in EMAILS if e["ground_truth"]["label"] == "archive"),
    },
    "by_priority": {
        "high": sum(1 for e in EMAILS if e["ground_truth"]["priority"] == "high"),
        "medium": sum(1 for e in EMAILS if e["ground_truth"]["priority"] == "medium"),
        "low": sum(1 for e in EMAILS if e["ground_truth"]["priority"] == "low"),
    },
}

if __name__ == "__main__":
    print(f"Email dataset: {EMAIL_STATS}")
