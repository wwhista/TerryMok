import json
import socket
import subprocess
import sys
import pytest


@pytest.mark.parametrize('argument,value', [('--review-limit', 'nan'),
    ('--short-limit', '2'), ('--min-messages', '0'), ('--consecutive', '0'),
    ('--samples', '1'), ('--seconds', 'nan')])
def test_cli_bad_configuration_has_usage_error(argument, value):
    process = subprocess.run([sys.executable, '-m', 'terrymok.collector',
        '--review-limit', '.4', '--short-limit', '.7', argument, value],
        capture_output=True, text=True, timeout=15)
    assert process.returncode == 2
    assert process.stdout == ''
    assert 'error:' in process.stderr
    assert 'Traceback' not in process.stderr


def test_cli_unavailable_server_emits_bounded_errors_and_fails():
    # Reserve a port without listening so another service cannot take it.
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        process = subprocess.run([sys.executable, '-m', 'terrymok.collector',
            '--port', str(reserved.getsockname()[1]), '--samples', '2', '--seconds', '.1',
            '--review-limit', '.4', '--short-limit', '.7'],
            capture_output=True, text=True, timeout=20)
    assert process.returncode == 1
    rows = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(rows) == 2
    assert all(row['status'] == 'collection_error' for row in rows)
    assert all(row['active_warnings'] == [] for row in rows)
    assert 'Traceback' not in process.stderr
