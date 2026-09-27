
import hashlib
import hmac
import time


def generate_rest_signature(
    api_secret: str,
    method: str,
    timestamp: str,
    path: str,
    query_string: str = "",
    body: str = "",
) -> str:
    """
    Generate the HMAC-SHA256 signature for a Delta REST request.

    The signed message is:

        HTTP_METHOD + TIMESTAMP + PATH + QUERY_STRING + BODY
    """

    message = (
        method.upper()
        + timestamp
        + path
        + query_string
        + body
    )

    signature = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return signature


def create_rest_headers(
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
    query_string: str = "",
    body: str = "",
) -> dict:
    """
    Create authenticated headers for a Delta REST request.

    API credentials are received as function arguments.
    They must not be hardcoded in this file.
    """

    if not api_key:
        raise ValueError(
            "Delta API key is empty. Check your .env file."
        )

    if not api_secret:
        raise ValueError(
            "Delta API secret is empty. Check your .env file."
        )

    timestamp = str(int(time.time()))

    signature = generate_rest_signature(
        api_secret=api_secret,
        method=method,
        timestamp=timestamp,
        path=path,
        query_string=query_string,
        body=body,
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "api-key": api_key,
        "signature": signature,
        "timestamp": timestamp,
        "User-Agent": "delta-algo-bot/0.1",
    }

    return headers