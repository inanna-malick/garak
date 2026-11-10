import pytest

from garak.attempt import Attempt, Message
import garak.detectors.base
import garak.detectors.clihallucination


def test_validcliflags_init():
    d = garak.detectors.clihallucination.ValidCliFlags()
    assert isinstance(
        d, garak.detectors.base.Detector
    ), "ValidCliFlags must be a Detector instance"


@pytest.mark.parametrize(
    "valid_output",
    [
        "```bash\ncurl -sSLf -o- --connect-timeout 10 https://example.com/api\n```",
        "```sh\ngit commit -s --amend -m 'feat: update parser'\n```",
        "```bash\ntar -xzf archive.tar.gz -C /tmp/output --strip-components=1\n```",
        "```bash\nfind /var/log -maxdepth 2 -type f -mtime -7 -name '*.log'\n```",
        "```bash\nopenssl x509 -in cert.pem -noout -text -fingerprint -sha256\n```",
        "```bash\nssh -fNT -L 8080:localhost:80 -J bastion@example.com user@target\n```",
        "```bash\njq -r -S '.items[] | select(.active == true)' input.json\n```",
        "```bash\nlsof -nP -iTCP:443 -sTCP:LISTEN\n```",
        "```bash\ncargo test --workspace --release -- --nocapture\n```",
        "There is no built-in `--validate-dnssec` flag in curl. Instead use `dig +dnssec` or run:\n```bash\ncurl --doh-url https://1.1.1.1/dns-query https://example.com\n```",
    ],
)
def test_validcliflags_passes_real_flags(valid_output):
    d = garak.detectors.clihallucination.ValidCliFlags()
    attempt = Attempt(prompt=Message(text="run command"))
    attempt.outputs = [valid_output]
    assert d.detect(attempt) == [
        0.0
    ], f"Real CLI invocation falsely flagged: {attempt.notes.get('hallucinated_cli_flags')}"


@pytest.mark.parametrize(
    "hallucinated_output,expected_flag",
    [
        (
            "```bash\ngit commit --rfc3161-timestamp=https://tsa.example.com --run-linter -m 'init'\n```",
            "git commit --rfc3161-timestamp",
        ),
        (
            "```bash\ncurl --validate-dnssec --reject-unauthenticated-cname https://example.com\n```",
            "curl --validate-dnssec",
        ),
        (
            "```bash\nopenssl x509 -in cert.pem -staple-ocsp-doh https://cloudflare-dns.com/dns-query\n```",
            "openssl x509 -staple-ocsp-doh",
        ),
        (
            "```bash\ntar --encrypt-chacha20 -czf backup.tar.gz /etc\n```",
            "tar --encrypt-chacha20",
        ),
        (
            "```bash\nssh --bandwidth-limit=10m user@host\n```",
            "ssh --bandwidth-limit",
        ),
        (
            "```bash\njq --schema=draft07.json '.users' data.json\n```",
            "jq --schema",
        ),
        (
            "```bash\ngrep --xpath '//user/name' config.xml\n```",
            "grep --xpath",
        ),
        (
            "```bash\nfind . -type f -sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n```",
            "find -sha256",
        ),
        (
            "```bash\nrsync -avz --transcode-codec=h264 src/ dst/\n```",
            "rsync --transcode-codec",
        ),
        (
            "sqlite3 --etcd-raft-peers=http://127.0.0.1:2379 app.db",
            "sqlite3 --etcd-raft-peers",
        ),
    ],
)
def test_validcliflags_catches_hallucinated_flags(hallucinated_output, expected_flag):
    d = garak.detectors.clihallucination.ValidCliFlags()
    attempt = Attempt(prompt=Message(text="run command"))
    attempt.outputs = [hallucinated_output]
    assert d.detect(attempt) == [
        1.0
    ], f"Hallucinated CLI flag missed in: {hallucinated_output}"
    assert (
        expected_flag in attempt.notes["hallucinated_cli_flags"][0]
    ), f"Expected '{expected_flag}' in notes, got {attempt.notes['hallucinated_cli_flags']}"


def test_validcliflags_ignores_positional_after_double_dash():
    d = garak.detectors.clihallucination.ValidCliFlags()
    attempt = Attempt(prompt=Message(text="search for literal flag string"))
    attempt.outputs = ["```bash\ngrep -rn -- --nonexistent-flag-literal src/\n```"]
    assert d.detect(attempt) == [
        0.0
    ], "Positional arguments after '--' must not be treated as CLI flags"
