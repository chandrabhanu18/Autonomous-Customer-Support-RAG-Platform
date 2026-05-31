import time
import sys
import json

import httpx

HEALTH_URL = 'http://localhost:8000/health'
QUERY_URL = 'http://localhost:8000/query'
EVAL_URL = 'http://localhost:8000/evaluate/{}'
FEEDBACK_URL = 'http://localhost:8000/feedback'
FEEDBACK_SUM_URL = 'http://localhost:8000/feedback/summary/{}'


def wait_for_health(timeout=120):
    client = httpx.Client()
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = client.get(HEALTH_URL, timeout=5.0)
            if r.status_code == 200:
                j = r.json()
                print('HEALTH', j)
                if j.get('status') == 'ok' and j.get('db_connected') is True:
                    return True
        except Exception as e:
            print('health check error:', e)
        time.sleep(1)
    return False


def run_smoke():
    client = httpx.Client(timeout=30.0)

    queries = [
        'How do I enable two-factor authentication?',
        'How do I export my project data as CSV?',
        "The webhook I set up isn't receiving any events",
    ]

    for q in queries:
        print('\n--- Running query:', q)
        resp = client.post(QUERY_URL, json={'query': q, 'top_k': 5})
        print('QUERY status', resp.status_code)
        try:
            payload = resp.json()
        except Exception:
            print('invalid json from /query:', resp.text)
            continue
        print(json.dumps(payload, indent=2))

        response_id = payload.get('response_id')
        if not response_id:
            print('no response_id, skipping eval/feedback')
            continue

        ev = client.post(EVAL_URL.format(response_id))
        print('EVAL', ev.status_code, ev.text)

        fb = client.post(FEEDBACK_URL, json={'response_id': response_id, 'rating': 4, 'comment': 'composer smoke test'})
        print('FEEDBACK', fb.status_code, fb.text)

        summary = client.get(FEEDBACK_SUM_URL.format(response_id))
        print('SUMMARY', summary.status_code, summary.text)


if __name__ == '__main__':
    ok = wait_for_health(timeout=180)
    if not ok:
        print('API did not become healthy in time')
        sys.exit(2)
    print('API healthy — running smoke tests')
    run_smoke()
    print('\nSmoke tests finished')
