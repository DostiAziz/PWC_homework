# UI walkthrough proposal

Status: client-flow direction clarified by the user; detailed interface and review rules remain proposals. Interactive previews are sample data, not application implementation.

## People and opening screens

The client experience is primary. A direct-question client opens a simple conversation screen, enters a question, and submits it. Submission triggers the workflow. Supported general questions receive an automatic answer with relevant sources; applicable cases pause for human review. An email sender follows the same business flow through the simulated email channel in version 1. The employee workspace is the internal interface for review and oversight.

## Employee flow

1. Open Inbox and select an enquiry. Show the sender, subject, original message, conversation history, language, and case status.
2. Client submission initiates processing. A Process enquiry button may remain as a demo/reprocessing control, but an employee click is not required for each client enquiry.
3. Display observable workflow activity, retrieved evidence, the suggested reply, and its citations. Show operational events and routing outcomes, not private model reasoning.
4. Supported general enquiries are answered automatically after the applicable checks. The earlier employee mockup's universal manual-release step is superseded by the user's clarified client flow.
5. Critical enquiries pause in Human review. The reviewer sees the original message, escalation reason, available evidence, and a draft if appropriate. Allow approval, editing, rejection, or human ownership. Edited content must pass the applicable checks; approval is associated with the current reply version.
6. An approved release produces one simulated outbox entry and updates the case. Taking human ownership without sending creates no outbox reply. Keep simulation labels visible.

## Direct-question flow

The customer opens a conversation screen and submits a question. Show progress followed by an automatic supported answer with source references, a clarification request, or a notice that a human review is pending. The client does not choose the route. The same backend workflow handles email and direct questions, but the customer does not receive internal reviewer controls. Recommend resuming the same conversation for follow-up messages and human responses. Do not promise a response deadline unless a service policy has established one.

## Preview boundaries

The interactive preview contains two entirely synthetic enquiries: an insurer asking about service areas and a report of possible confidential-document exposure. Responses and workflow events are fixed sample data. No Ollama calls, retrieval, persisted state, security enforcement, or email delivery run in this preview. Review criteria are proposed demo rules, not verified PwC internal policy.

The earlier employee preview demonstrates manual draft release and reviewer actions. Its routine release behavior is superseded by the user clarification above. The client preview demonstrates a general question answered automatically, a sensitive case awaiting review, and a specialist's follow-up in the same conversation. Its example selector and advance-to-reviewed-state control are demonstration controls outside the depicted client interface. Fixed case references are synthetic; the implementation must display references only after successful case creation.

Open decisions: mandatory-review categories and checks, reviewer permissions, client identity and conversation access, notification mechanism, and the final Streamlit layout.
