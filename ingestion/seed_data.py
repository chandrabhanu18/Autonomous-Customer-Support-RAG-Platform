from __future__ import annotations

from typing import Callable


def _ensure_min_words(base: str, appendix: str, minimum: int = 320) -> str:
    content = base.strip()
    appendix_text = appendix.strip()
    while len(content.split()) < minimum:
        content = f"{content}\n\n{appendix_text}"
    return content


def _doc(doc_id: str, title: str, base: str, appendix: str, tags: list[str]) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "content": _ensure_min_words(base, appendix),
        "source_url": f"https://nexora.example.com/support/{doc_id}",
        "metadata": {"tags": tags, "product": "nexora"},
    }


def get_seed_documents() -> list[dict]:
    documents: list[dict] = []

    documents.append(
        _doc(
            "doc_001",
            "Getting Started with Nexora",
            """
Nexora is designed to help teams plan projects, track work, and keep stakeholders aligned without forcing them through a complicated setup. New customers should begin by creating a workspace, inviting a small pilot team, and adding one active project so the product can immediately demonstrate value. The first week is best spent configuring the workspace name, selecting the primary time zone, and deciding which notifications should be visible to everyone versus only project owners. The onboarding flow also introduces the template library, which gives teams a fast path to their first project board.

After the workspace is created, administrators should review the default roles and determine whether the team needs contributors, managers, or observers. A good practice is to start with a simple permissions model, then tighten access for sensitive projects later. Nexora supports a gradual rollout, so the support team often recommends connecting one integration at a time and validating that tasks, comments, and status updates behave as expected. The dashboard includes a guided checklist that tracks progress and highlights missing steps such as profile setup, team invites, and notification preferences. Completing those steps reduces confusion and gives users a predictable starting point.
""",
            """
A successful launch depends on helping users understand where to find common actions. The left sidebar contains the project list, the center panel shows work items and recent activity, and the right panel exposes context such as assignees, deadlines, and linked files. During onboarding, it helps to explain that search is global across projects, comments, and documents, while filters are local to the current view. Support engineers should remind new teams that they can duplicate a project template, edit the workflow stages, and save the customized version for later use.

If a workspace feels empty after setup, the usual cause is that no members have accepted their invitations or no project has been assigned to the default board. In that case, verify the invite emails, check the user roles, and confirm that the project is published rather than remaining in draft mode. The safest rollout is small, measurable, and explicit. Teams should review the first project together, add a short list of tasks, and confirm that comments, mentions, and notifications are flowing to the right people before expanding to the rest of the organization.
""",
            ["onboarding", "workspace", "getting-started"],
        )
    )

    documents.append(
        _doc(
            "doc_002",
            "Managing Team Members and Permissions",
            """
Team management in Nexora starts with a simple principle: invite only the people who need access, then expand permissions deliberately as the workspace matures. Administrators can add members from the workspace settings panel, assign them to teams, and choose whether they can create projects, manage billing, or only view assigned work. Each role is designed to reduce accidental changes while still letting people contribute. For example, project managers typically need edit access to tasks and workflows, while finance staff may only need billing visibility.

Permissions are layered across the workspace, project, and item levels. That means a user can be a workspace member but still have restricted access to specific projects. This matters when a customer runs multiple client workspaces or keeps confidential programs separate from general operations. When a support request mentions someone unable to see a project, the first checks should be membership, project assignment, and whether the project has been archived or hidden. A common mistake is to assume a user role alone explains access; in practice, the project-level sharing settings often determine what appears in the sidebar.
""",
            """
Nexora also supports deactivating members without deleting their history. Deactivation preserves task assignments, comments, and audit records, which is useful when staff leave or move to another department. If a customer needs to remove a person completely, support should advise exporting any necessary records first because the audit trail is designed to stay intact for compliance purposes. Reassigning work before deactivation prevents incomplete handoffs and makes it easier for the remaining team to understand what changed.

Two-factor authentication, email verification, and session controls are part of the overall account management story, because unauthorized access often looks like a permissions problem. If a user cannot sign in, check whether the invitation was accepted, whether the email address is correct, and whether the account is locked after repeated failures. Managers can also review access logs to determine whether a member is using the right workspace. Clear permission design reduces support load, so teams should document who can invite users, who can approve access changes, and who owns the security configuration.
""",
            ["permissions", "team-members", "access-control"],
        )
    )

    documents.append(
        _doc(
            "doc_003",
            "Billing and Subscription Plans",
            """
Nexora offers monthly and annual subscription plans that scale by team size and feature access. Customers usually begin on a trial or starter tier and upgrade once they need additional automation, integrations, or advanced reporting. Billing is handled from the account settings area, where administrators can review current plan details, add or update payment methods, download invoices, and see the next renewal date. When a customer asks about price changes, support should identify the current plan, the billing cycle, and whether taxes or regional adjustments apply.

Invoices are generated automatically after each billing event and remain available in the billing history for later reference. If a payment fails, the system retries according to the configured schedule and notifies the account owner. A failed payment does not immediately remove access, but repeated failures can place the subscription in a limited state. That state is intentionally designed to give customers time to update payment details without losing active work. Support conversations about billing should include the invoice number, the amount charged, and any visible status notes from the billing page.
""",
            """
Refunds, cancellations, and plan downgrades are all governed by the subscription policy attached to the account. Annual plans may continue until the end of the paid period, while monthly plans typically stop at the next renewal boundary. If a customer cancels because they no longer need the product, support should explain what data remains available, what export options exist, and whether any prorated credits apply. The product also provides a usage summary so administrators can see which limits are close to being exceeded and whether it makes sense to move to a larger plan.

When billing questions become urgent, support should verify the exact account before taking action, because many customers manage more than one workspace. The safest workflow is to confirm the workspace name, the contact email, the subscription tier, and the last successful charge before modifying anything. This reduces mistakes, especially when the customer is requesting a refund or a card update. Clear billing records and explicit renewal notices make it easier for finance teams to audit changes and for support agents to explain what happened without guessing.
""",
            ["billing", "subscription", "invoices"],
        )
    )

    documents.append(
        _doc(
            "doc_004",
            "Integrations: Slack, GitHub, and Jira",
            """
Nexora integrates with Slack, GitHub, and Jira so teams can keep work moving without duplicating updates across tools. Slack integration is usually the first connection because it lets project events appear in dedicated channels, helping teams see status changes and mentions as they happen. Administrators can authorize the workspace, choose which channels receive notifications, and decide whether the integration posts every activity or only high-priority events. The Slack setup wizard checks permissions carefully because channel access and message posting require explicit approval.

GitHub integration is built for engineering teams that want commits, pull requests, and issue references connected to project items. Once the repository is connected, Nexora can show development activity in the project timeline and make it easier to trace work from planning to release. Jira support follows a similar pattern, but the focus is on issue synchronization, status mapping, and bidirectional updates. Teams should define which system is the source of truth before turning on synchronization, otherwise a ticket can bounce between tools with conflicting statuses.
""",
            """
When an integration fails, support should inspect authorization first and event delivery second. Expired tokens, revoked permissions, and mismatched workspace IDs are the most common causes. For Slack, the channel may exist but the bot may lack permission to post. For GitHub, the webhook may be active but filtered to the wrong repository. For Jira, the field mapping might not match the customer workflow, so statuses appear to sync even though important metadata is missing. These problems are usually fixed by reconnecting the account and reviewing the integration logs.

Customers should also understand that integrations are scoped. A workspace can connect one Slack workspace, several GitHub repositories, and multiple Jira projects, but each connection needs clear ownership. Support teams often recommend naming conventions and a test channel before rolling the integration out broadly. That makes it easier to verify alerts, mentions, and issue updates without spamming production channels. Good integration hygiene reduces noise and helps teams trust that Nexora is moving the right information to the right external system.
""",
            ["slack", "github", "jira", "integration"],
        )
    )

    documents.append(
        _doc(
            "doc_005",
            "Project Templates and Workflows",
            """
Project templates let teams start with a proven structure instead of building everything from zero. In Nexora, a template can include stages, default assignees, custom fields, automation rules, and example tasks. Many customers use templates for recurring processes such as onboarding, product launches, and customer implementation. The most useful templates are not overly complex; they capture the essential steps while leaving room for the team to adapt the workflow to its own pace. Administrators can create templates from scratch or duplicate an existing project and strip it down to a reusable core.

Workflows define how an item moves through the system. A workflow might begin in "Backlog," move to "In Progress," then transition through review, approval, and completion. Teams can rename stages, reorder them, and add rules so that a task cannot advance until required fields are filled in. This is especially helpful when different departments need different approval steps. Support should explain that templates and workflows are related but not identical: the template defines the starting structure, while the workflow controls how work behaves after the project begins.
""",
            """
Customers often ask whether they can use custom templates for different teams, and the answer is yes. A marketing template may include campaign briefs and launch dates, while an engineering template may include code review and deployment checkpoints. The most effective approach is to design templates around repeatable business outcomes, not around a single individual’s preferences. If a workflow becomes too rigid, teams can end up bypassing it entirely, so the best templates are flexible enough to support real-world exceptions without losing visibility.

When troubleshooting template issues, support should check whether the project was created from the correct template version and whether stage automation rules were copied correctly. Some users mistakenly edit the live project instead of the saved template, which makes future projects inconsistent. A good practice is to name templates clearly, document their purpose, and review them quarterly. That keeps process design aligned with how the organization actually works and prevents stale workflows from slowing teams down.
""",
            ["templates", "workflow", "projects"],
        )
    )

    documents.append(
        _doc(
            "doc_006",
            "Notifications and Alert Settings",
            """
Notifications in Nexora are meant to keep users informed without overwhelming them. Each user can configure email, in-app, and channel-based alerts for mentions, task assignments, due date reminders, and workflow changes. Workspace owners often start with a broader set of notifications, then narrow them once the team understands which events actually matter. The product also supports quiet hours so people are not interrupted by late-night updates or weekend activity. This helps support teams explain why a user may not receive a message immediately even when the underlying event occurred.

Alert settings are divided between personal preferences and workspace rules. Personal preferences let a user decide which events appear in their inbox, while workspace rules determine which events are generated at all. That distinction matters when a manager says the team is not seeing alerts: the issue could be that the event never fired, that the rule only applies to specific projects, or that a member muted the notification category. Nexora includes a notification preview so users can test their settings before relying on them for critical work.
""",
            """
For support, it is important to ask when the alert should have arrived, which channel was expected, and whether any filters or mobile settings were active. Email notifications can be delayed by external providers, while push notifications can fail if the device is offline or the app is not allowed to send them. In-app alerts are typically the most reliable because they are generated directly in the product. If a user reports that a project update was missed, check whether they were assigned to the item, whether the event was set to notify only watchers, and whether they recently changed their notification profile.

Teams should review alerts regularly because a noisy notification system creates fatigue and leads to ignored messages. Support agents can recommend limiting high-volume events to digest emails or summary notifications. The goal is to make sure urgent changes stand out while routine activity stays visible but unobtrusive. When the settings are tuned well, the team spends less time chasing status updates and more time finishing work.
""",
            ["notifications", "alerts", "reminders"],
        )
    )

    documents.append(
        _doc(
            "doc_007",
            "Data Export and Backup",
            """
Nexora provides export tools so customers can keep backups of project data, preserve audit records, and move information into other systems when needed. Exports are available from the workspace administration area and can include tasks, comments, custom fields, attachments metadata, and activity history depending on the selected scope. The system usually generates CSV files for tabular data and JSON for more structured payloads. Support teams should explain that exports are asynchronous for large workspaces, meaning the file may take a few minutes to appear once the request is submitted.

Backup behavior is different from export behavior. Exports are user-initiated snapshots, while backups are managed by the platform to protect against loss. Customers sometimes assume an export is the same as a full recovery point, but that is not always true because some system-level configuration or transient records may not be included. The platform keeps scheduled backups so administrators can restore data after operational incidents, but those restores are typically coordinated by support or operations teams rather than the end user.
""",
            """
A common support request is to export project data as CSV for analysis in a spreadsheet or business intelligence tool. In that case, confirm the date range, the workspace, and whether the customer wants task-level or activity-level rows. If an export seems incomplete, the cause is often a filter that excludes archived items, a permissions issue that hides private projects, or a large file that must finish processing before download. The user should also make sure the browser did not block the download or discard the pop-up notification that the file was ready.

When backup or export questions involve compliance, the safest response is to be precise about what the platform guarantees and what it does not. Support should avoid assuming that a backup contains every possible object unless the customer has confirmed the backup policy for that workspace. Clear communication about retention windows, export scope, and recovery steps helps customers trust the system when they need to move or restore their data. It also reduces surprises during audits or migration projects.
""",
            ["export", "backup", "csv", "data"],
        )
    )

    documents.append(
        _doc(
            "doc_008",
            "Two-Factor Authentication Setup",
            """
Two-factor authentication adds an extra security layer to Nexora accounts by requiring something the user knows and something the user has. The setup flow begins in account security settings, where a user can choose an authenticator app or another supported second factor. Once enabled, the system prompts the user to scan a QR code or enter a setup key, then verify the first generated code before marking the account as protected. Support should remind customers to store recovery codes in a safe place because those codes may be the only way to regain access if the device is lost.

Nexora encourages administrators to require two-factor authentication for sensitive workspaces, especially when team members can see billing details or manage integrations. If a user is already signed in and wants to enable 2FA, the platform checks password confirmation before making the change. This helps prevent an attacker who briefly gained access from turning off security controls. The account activity page also shows recent authentication events so support can confirm whether a user has already activated 2FA or whether the setup was interrupted midway through the process.
""",
            """
If a customer cannot complete setup, the most common problems are inaccurate device time, an expired setup session, or a mismatched recovery code. In some cases the user loses the QR code before the authenticator app has saved the account entry. When that happens, the correct fix is usually to restart the setup process rather than repeatedly entering invalid codes. Support should also ask whether the customer uses a company-managed mobile device, because device policies can block camera access or prevent the authenticator app from storing new accounts.

When a user says they forgot their password and cannot log in, support should think about the whole authentication path, not just the password reset flow. The account may be locked after too many attempts, the user may be entering an old email address, or the workspace may require 2FA before allowing access. The help article should explain recovery codes, email verification, and how administrators can reset security settings when absolutely necessary. Clear, step-by-step guidance prevents users from getting locked out while still preserving the account protections the organization expects.
""",
            ["two-factor", "2fa", "security", "authentication"],
        )
    )

    documents.append(
        _doc(
            "doc_009",
            "API Access and Webhooks",
            """
Nexora exposes an API so teams can integrate the platform with internal tools, automate task management, and build custom reporting pipelines. API access is controlled by personal or workspace tokens depending on the customer plan and security policy. Developers should store tokens securely, rotate them regularly, and limit them to the minimum scope required for the job. The API documentation explains authentication headers, pagination, filtering, and the objects returned by each endpoint. Support should encourage customers to test requests in a safe environment before using them in production workflows.

Webhooks extend the API by sending outbound event notifications whenever important changes occur. A webhook might fire when a task is created, updated, completed, or commented on. Administrators can subscribe to specific event types and choose a target URL for delivery. Successful integration depends on the receiving endpoint acknowledging the payload quickly. If the destination returns repeated errors or takes too long to respond, Nexora retries delivery according to its retry policy and records the attempt in the integration log.
""",
            """
When a webhook is not receiving events, support should verify the subscription, the event filter, the target URL, and whether the remote server has started returning errors. Many customers assume the webhook failed inside Nexora when the real issue is an unreachable endpoint or an authentication failure on the receiving system. The logs usually show whether payload delivery happened, whether the response code was acceptable, and whether the retry mechanism was triggered. That makes it easier to determine whether the problem is network-related, configuration-related, or tied to the remote service itself.

API and webhook questions often overlap with security reviews. Customers want to know how long tokens last, whether they can be scoped to a single workspace, and how to audit calls that were made with a given credential. Good support answers should mention token rotation, secret storage, and the fact that webhook signatures help verify that a message truly came from Nexora. Teams that treat API access as a governed integration rather than an ad hoc shortcut have fewer outages and fewer surprises when they need to debug production automation.
""",
            ["api", "webhooks", "tokens", "integrations"],
        )
    )

    documents.append(
        _doc(
            "doc_010",
            "Troubleshooting Common Errors",
            """
Support for a project management platform always involves a mix of browser issues, permission problems, integration failures, and data sync confusion. Nexora documents the most common error patterns so support agents can start from the right branch of the troubleshooting tree. When a user reports a generic error, the first step is to capture the exact message, the time it occurred, the project affected, and whether the issue happened once or repeatedly. Those details help determine whether the problem belongs to authentication, billing, notifications, or a deeper service outage.

Many errors are caused by stale sessions or browser cache problems. If the user is stuck on an old page, support can ask them to hard refresh, try an incognito window, or sign out and back in. If the behavior persists, the next step is usually to check whether the user has the proper role, whether the project is archived, and whether an integration or webhook is returning a failure. The goal is to separate front-end display issues from actual data or service errors.
""",
            """
A useful support habit is to look for patterns across multiple users. If only one person sees the issue, the cause is often local to their browser, permissions, or device settings. If many users report the same behavior, the problem may involve a shared integration, an API dependency, or a platform-wide incident. Nexora recommends capturing error codes when available because codes are far easier to search than vague descriptions. For example, a code tied to authentication should trigger account checks, while a code tied to a webhook delivery problem should trigger integration logs.

The best troubleshooting answers are calm, specific, and action oriented. They explain what to collect, what to try first, and when to escalate. Even when the root cause is not obvious, support can often narrow the search by asking about the last successful action, recent changes, and whether the error appears across devices. A structured approach saves time and helps customers feel that the issue is being handled methodically rather than guessed at.
""",
            ["errors", "troubleshooting", "debugging"],
        )
    )

    return documents


SEED_DOCUMENTS = get_seed_documents()
