"""Offline integration: real loopback HTTP, synthetic scores, no corpus download."""
import hashlib
import json
import socket
import threading
import time
from urllib.request import Request, urlopen

import joblib
import uvicorn

from terrymok.api import create_app
from terrymok.collector import Collector, poll


class SyntheticModel:
    classes_ = [0, 1, 2]

    def decision_function(self, texts):
        return [[2, 0, 0] if len(text.split()) > 20 else [0.1, 0, 0] for text in texts]


def test_http_monitoring_warning_and_recovery(tmp_path):
    model = tmp_path / 'model.joblib'
    joblib.dump({'pipeline': SyntheticModel(), 'labels': ['a', 'b', 'c']}, model)
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'labels': ['a', 'b', 'c'], 'selected': {'threshold': 1},
        'score': 'largest minus second-largest decision score',
        'rule': 'gap >= threshold; null threshold means review all'}))
    # Keep the socket reserved throughout startup, avoiding free-port races.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(model, policy),
            host='127.0.0.1', port=port, log_level='error', access_log=False,
            lifespan='on', ws='none'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            assert server.started, 'HTTP server failed to start'
            base = f'http://127.0.0.1:{port}'
            def fetch():
                with urlopen(base + '/metrics', timeout=3) as response:
                    return json.load(response)
            def send(text):
                request = Request(base + '/predict', data=json.dumps({'texts': [text] * 2}).encode(),
                                  headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=3) as response:
                    assert response.status == 200
                    return json.load(response)
            samples = poll(Collector(.4, .7, min_messages=2, consecutive=2), fetch, 5, 0.05)
            assert next(samples)['status'] == 'baseline'
            send('word ' * 25)
            assert next(samples)['active_warnings'] == []
            assert all(p['review_required'] for p in send('short')['predictions'])
            assert next(samples)['active_warnings'] == []
            send('short')
            active = next(samples)
            assert active['active_warnings'] == ['review', 'short_messages'], active
            send('word ' * 25)
            recovered = next(samples)
            assert recovered['active_warnings'] == []
            assert recovered['interval']['messages'] == 2
            metrics = fetch()
            assert metrics['predictions']['messages'] == 8
            assert metrics['predictions']['review_messages'] == 4
            assert metrics['completed'] == 4
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), 'HTTP server did not shut down'
