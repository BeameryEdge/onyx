import io
import ipaddress
import socket
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Tuple
from urllib.parse import urljoin, urlparse

import asyncio
import requests
from bs4 import BeautifulSoup
from oauthlib.oauth2 import BackendApplicationClient
from playwright.async_api import async_playwright, TimeoutError
from requests_oauthlib import OAuth2Session
from urllib3.exceptions import MaxRetryError

from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.app_configs import WEB_CONNECTOR_OAUTH_CLIENT_ID
from onyx.configs.app_configs import WEB_CONNECTOR_OAUTH_CLIENT_SECRET
from onyx.configs.app_configs import WEB_CONNECTOR_OAUTH_TOKEN_URL
from onyx.configs.app_configs import WEB_CONNECTOR_VALIDATE_URLS
from onyx.configs.constants import DocumentSource
from onyx.connectors.exceptions import ConnectorValidationError, CredentialExpiredError, InsufficientPermissionsError, UnexpectedValidationError
from onyx.connectors.interfaces import GenerateDocumentsOutput, LoadConnector
from onyx.connectors.models import Document, Section
from onyx.file_processing.extract_file_text import read_pdf_file
from onyx.file_processing.html_utils import web_html_cleanup
from onyx.utils.logger import setup_logger
from onyx.utils.sitemap import list_pages_for_site
from shared_configs.configs import MULTI_TENANT

logger = setup_logger()

WEB_CONNECTOR_MAX_SCROLL_ATTEMPTS = 20
IFRAME_TEXT_LENGTH_THRESHOLD = 700
JAVASCRIPT_DISABLED_MESSAGE = "You have JavaScript disabled in your browser"


class WEB_CONNECTOR_VALID_SETTINGS(str, Enum):
    RECURSIVE = "recursive"
    SINGLE = "single"
    SITEMAP = "sitemap"
    UPLOAD = "upload"


async def protected_url_check(url: str) -> None:
    """Ensures URL is valid and publicly accessible."""
    if not WEB_CONNECTOR_VALIDATE_URLS:
        return

    parse = urlparse(url)
    if parse.scheme not in {"http", "https"}:
        raise ValueError("URL must be of scheme http:// or https://")

    if not parse.hostname:
        raise ValueError("URL must include a hostname")

    try:
        info = socket.getaddrinfo(parse.hostname, None)
    except socket.gaierror as e:
        raise ConnectionError(f"DNS resolution failed for {parse.hostname}: {e}")

    for address in info:
        ip = address[4][0]
        if not ipaddress.ip_address(ip).is_global:
            raise ValueError(f"Non-global IP detected: {ip}, skipping page {url}.")


async def check_internet_connection(url: str) -> None:
    """Checks if the target URL is reachable."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            response = await page.goto(url, timeout=6000, wait_until="domcontentloaded")
            if response is None:
                raise Exception(f"Failed to fetch {url} - No response received")

            if response.status >= 400:
                raise Exception(f"HTTP Error {response.status} for {url}")

        finally:
            await browser.close()


async def start_playwright() -> Tuple[Any, Any]:
    """Initializes and starts Playwright."""
    try:
        p = await async_playwright().start()
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
        return p, context
    except Exception as e:
        raise RuntimeError(f"Error initializing Playwright: {e}")


class WebConnector(LoadConnector):
    def __init__(
        self,
        base_url: str,
        web_connector_type: str = WEB_CONNECTOR_VALID_SETTINGS.RECURSIVE.value,
        mintlify_cleanup: bool = True,
        batch_size: int = INDEX_BATCH_SIZE,
        scroll_before_scraping: bool = False,
        **kwargs: Any,
    ) -> None:
        self.mintlify_cleanup = mintlify_cleanup
        self.batch_size = batch_size
        self.recursive = False
        self.scroll_before_scraping = scroll_before_scraping
        self.web_connector_type = web_connector_type

        if web_connector_type == WEB_CONNECTOR_VALID_SETTINGS.RECURSIVE.value:
            self.recursive = True
            self.to_visit_list = [base_url]
        elif web_connector_type == WEB_CONNECTOR_VALID_SETTINGS.SINGLE.value:
            self.to_visit_list = [base_url]
        elif web_connector_type == WEB_CONNECTOR_VALID_SETTINGS.SITEMAP:
            self.to_visit_list = list_pages_for_site(base_url)
        else:
            raise ValueError("Invalid Web Connector Type")

    async def load_from_state(self) -> GenerateDocumentsOutput:
        visited_links = set()
        to_visit = self.to_visit_list
        content_hashes = set()

        if not to_visit:
            raise ValueError("No URLs to visit")

        base_url = to_visit[0]
        doc_batch = []
        last_error = None

        p, context = await start_playwright()

        while to_visit:
            initial_url = to_visit.pop()
            if initial_url in visited_links:
                continue
            visited_links.add(initial_url)

            try:
                await protected_url_check(initial_url)
            except Exception as e:
                last_error = f"Invalid URL {initial_url} due to {e}"
                logger.warning(last_error)
                continue

            logger.info(f"Visiting {initial_url}")
            try:
                await check_internet_connection(initial_url)
                page = await context.new_page()
                await asyncio.sleep(5)  # Prevent bot detection
                response = await page.goto(initial_url, timeout=30000)
                await page.wait_for_load_state("domcontentloaded", timeout=30000)

                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")

                if self.recursive:
                    internal_links = {
                        urljoin(initial_url, a.get("href"))
                        for a in soup.find_all("a", href=True)
                        if a.get("href").startswith("/")
                    }
                    to_visit.extend(link for link in internal_links if link not in visited_links)

                parsed_html = web_html_cleanup(soup, self.mintlify_cleanup)
                hashed_text = hash((parsed_html.title, parsed_html.cleaned_text))
                if hashed_text in content_hashes:
                    logger.info(f"Skipping duplicate {initial_url}")
                    continue
                content_hashes.add(hashed_text)

                doc_batch.append(
                    Document(
                        id=initial_url,
                        sections=[Section(link=initial_url, text=parsed_html.cleaned_text)],
                        source=DocumentSource.WEB,
                        semantic_identifier=parsed_html.title or initial_url,
                        metadata={},
                    )
                )

                await page.close()

            except Exception as e:
                last_error = f"Failed to fetch {initial_url}: {e}"
                logger.exception(last_error)

                await context.close()
                p, context = await start_playwright()
                continue

            if len(doc_batch) >= self.batch_size:
                yield doc_batch
                doc_batch = []

        if doc_batch:
            yield doc_batch

        await context.close()
        if last_error:
            raise RuntimeError(last_error)


if __name__ == "__main__":
    connector = WebConnector("https://docs.onyx.app/")
    asyncio.run(connector.load_from_state())
