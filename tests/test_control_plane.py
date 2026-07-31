import pytest

from app.control_plane import parse_job_metadata


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"callId":"call-1","swarmId":"swarm-1","direction":"outbound"}',
            ("call-1", "swarm-1", "outbound"),
        ),
        ("", ("", "", "inbound")),
        ("not-json", ("", "", "inbound")),
    ],
)
def test_parse_job_metadata_is_safe(raw: str, expected: tuple[str, str, str]) -> None:
    metadata = parse_job_metadata(raw)
    assert (metadata.call_id, metadata.swarm_id, metadata.direction) == expected
