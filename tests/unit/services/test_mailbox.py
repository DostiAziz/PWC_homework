from pathlib import Path

from pwc_support.adapters.simulated_mailbox import SimulatedMailbox
from pwc_support.domain.models import Channel


def test_simulated_mailbox_preserves_thread_and_outbox(tmp_path: Path) -> None:
    mailbox = SimulatedMailbox(tmp_path / "mailbox.json")

    message = mailbox.receive(
        sender="client@example.test", subject="Services", body="What does PwC offer?"
    )
    sent = mailbox.send(
        thread_id=message.provider_thread_id,
        recipient="client@example.test",
        subject="Re: Services",
        body="Please see our public services information.",
    )

    assert message.channel is Channel.SIMULATED_EMAIL
    assert sent.provider_thread_id == message.provider_thread_id
    assert len(mailbox.outbox()) == 1
