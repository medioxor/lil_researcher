from typing import List, Optional
from urllib.parse import urlparse
from playwright.async_api import (
    async_playwright,
)
from logging import Logger, getLogger, basicConfig, INFO


class Browser:
    def __init__(
        self,
        headless: bool = True,
        domain_whitelist: Optional[List[str]] = None,
    ):
        """
        Args:
            headless: Whether to run the browser in headless mode
            domain_whitelist: Optional list of allowed domains (e.g., ['example.com', 'api.example.org'])
        """
        self.headless: bool = headless
        self.domain_whitelist = domain_whitelist or []
        self.playwright = None
        self.browser = None
        self.context = None
        self._last_search_position = 0
        self._last_search_index = 0
        self._current_viewport_content = ""
        self.page = None
        self.logger: Logger = getLogger("browser")
        basicConfig(
            level=INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def __is_domain_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the whitelist."""
        if not self.domain_whitelist:
            return True  # If no whitelist is specified, allow all domains

        try:
            domain = urlparse(url).netloc
            # Exact match only - don't allow subdomains unless explicitly specified
            return domain in self.domain_whitelist
        except Exception:
            return False

    async def __route_handler(self, route, request):
        """Handle route requests and block if domain not in whitelist."""
        if self.__is_domain_allowed(request.url):
            await route.continue_()
        else:
            await route.abort()

    async def start(self) -> None:
        """Start the browser and create a new context and page."""
        self.logger.info("Creating browser and playwright resources")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1024, "height": 768}
        )
        self.page = await self.context.new_page()

        # Set up route handler to block non-whitelisted domains
        if self.domain_whitelist:
            await self.context.route("**/*", self.__route_handler)

    async def close(self) -> None:
        """Close browser and playwright resources."""
        self.logger.info("Closing browser and playwright resources")
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str, wait_until: str = "networkidle") -> bool:
        """
        Navigate to a URL and wait for the navigation to complete.

        Args:
            url: The URL to navigate to
            wait_until: Navigation wait criteria ('domcontentloaded', 'load', 'networkidle')

        Returns:
            bool: True if navigation was successful, False otherwise
        """
        if not self.page:
            self.logger.warning("No page available, can't navigate")
            return False

        try:
            # Check if domain is allowed before navigating
            if not self.__is_domain_allowed(url):
                self.logger.warning(f"Navigation to {url} blocked by domain whitelist")
                return False

            self.logger.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until=wait_until)
            return True
        except Exception as e:
            self.logger.error(f"Error navigating to {url}: {str(e)}")
            return False

    async def page_down(self) -> str:
        """
        Scroll down to the next viewport, showing completely new content.

        Returns:
            str: Content of the viewport after scrolling, "END OF PAGE REACHED" if at bottom, or empty string if error occurred
        """
        if not self.page:
            self.logger.warning("No page available, can't scroll")
            return ""

        try:
            # Check if we can scroll further down
            can_scroll = await self.page.evaluate("""
                () => {
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    const scrollHeight = Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight
                    );
                    const clientHeight = document.documentElement.clientHeight;
                    return scrollTop + clientHeight < scrollHeight;
                }
            """)

            if not can_scroll:
                self.logger.info("Already at the bottom of the page")
                return "END OF PAGE REACHED"

            # Get viewport height and scroll exactly one full viewport
            viewport_height = await self.page.evaluate(
                "document.documentElement.clientHeight"
            )

            # Scroll down by exactly one viewport
            await self.page.evaluate(f"window.scrollBy(0, {viewport_height})")
            await self.page.wait_for_timeout(100)  # Small wait for scroll to complete

            self.logger.info(f"Scrolled down by {viewport_height} pixels")

            # Get and return the new viewport content
            return await self.get_viewport_content()

        except Exception as e:
            self.logger.error(f"Error scrolling down: {str(e)}")
            return ""

    async def page_up(self) -> str:
        """
        Scroll up to the previous viewport, showing completely new content.

        Returns:
            str: Content of the viewport after scrolling, "TOP OF PAGE REACHED" if at top, or empty string if error occurred
        """
        if not self.page:
            self.logger.warning("No page available, can't scroll")
            return ""

        try:
            # Check if we can scroll further up
            can_scroll = await self.page.evaluate("""
                () => {
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    return scrollTop > 0;
                }
            """)

            if not can_scroll:
                self.logger.info("Already at the top of the page")
                return "TOP OF PAGE REACHED"

            # Get viewport height and scroll exactly one full viewport
            viewport_height = await self.page.evaluate(
                "document.documentElement.clientHeight"
            )

            # Scroll up by exactly one viewport
            await self.page.evaluate(f"window.scrollBy(0, -{viewport_height})")
            await self.page.wait_for_timeout(100)  # Small wait for scroll to complete

            self.logger.info(f"Scrolled up by {viewport_height} pixels")

            # Get and return the new viewport content
            return await self.get_viewport_content()

        except Exception as e:
            self.logger.error(f"Error scrolling up: {str(e)}")
            return ""

    async def find_text(self, search_text: str) -> Optional[str]:
        """
        Search for text in the page, scrolling through it until found or end is reached.
        Returns the viewport content where the text was found.

        Args:
            search_text: The text to search for

        Returns:
            Optional[str]: Viewport content containing the text, or None if not found
        """
        if not self.page:
            self.logger.warning("No page available, can't search for text")
            return None

        try:
            # First check if text is in current viewport
            content = await self.get_viewport_content()
            if search_text.lower() in content.lower():
                self.logger.info(f"Text '{search_text}' found in current viewport")
                return content

            # Scroll through the page looking for the text
            # First reset to top of page
            await self.page.evaluate("window.scrollTo(0, 0)")
            await self.page.wait_for_timeout(100)

            while True:
                # Search for text in current viewport
                content = await self.get_viewport_content()
                if search_text.lower() in content.lower():
                    self.logger.info(f"Text '{search_text}' found after scrolling")
                    return content

                # Try scrolling down to next viewport
                scroll_result = await self.page_down()
                if scroll_result == "END OF PAGE REACHED":
                    # Reached the bottom without finding the text
                    break

            self.logger.warning(f"Text '{search_text}' not found on page")
            return None
        except Exception as e:
            self.logger.error(f"Error searching for text: {str(e)}")
            return None

    async def find_text_next(self, search_text: str) -> Optional[str]:
        """
        Search for the next occurrence of text in the page, methodically checking all
        occurrences in the current viewport before moving to the next section.
        Handles viewport overlap to ensure each occurrence is only returned once.

        Args:
            search_text: The text to search for

        Returns:
            Optional[str]: Viewport content containing the text, or None if not found
        """
        if not self.page:
            self.logger.warning("No page available, can't search for text")
            return None

        try:
            search_text_lower = search_text.lower()

            # Initialize or refresh tracking variables if needed
            if not hasattr(self, "_found_positions"):
                self._found_positions = (
                    set()
                )  # Track absolute positions of found occurrences

            if not self._current_viewport_content:
                self._current_viewport_content = await self.get_viewport_content()
                self._last_search_position = -1
                self._last_search_index = 0

            # Loop to handle both current viewport search and scrolling for more content
            while True:
                # Get current scroll position for absolute positioning
                scroll_y = await self.page.evaluate("window.scrollY")

                # Look for next match in current viewport
                content_lower = self._current_viewport_content.lower()

                found_unique_match = False
                while not found_unique_match:
                    next_pos = content_lower.find(
                        search_text_lower, self._last_search_position + 1
                    )

                    if next_pos == -1:
                        # No more matches in current viewport
                        break

                    # Calculate absolute position of match in the document
                    # We use an approximation since we can't get exact pixel positions from text
                    absolute_pos = scroll_y + next_pos

                    # Check if this is a new unique match (with tolerance for small differences)
                    is_duplicate = any(
                        abs(absolute_pos - pos) < 50 for pos in self._found_positions
                    )

                    if not is_duplicate:
                        # Found a new unique match
                        self._found_positions.add(absolute_pos)
                        self._last_search_position = next_pos
                        self._last_search_index += 1
                        self.logger.info(
                            f"Found occurrence #{self._last_search_index} of '{search_text}'"
                        )
                        return self._current_viewport_content
                    else:
                        # Skip duplicate match and continue searching
                        self._last_search_position = next_pos

                # No more unique matches in current viewport, need to scroll for more content
                viewport_height = await self.page.evaluate(
                    "document.documentElement.clientHeight"
                )
                scroll_step = int(
                    viewport_height * 0.5
                )  # 50% overlap for better content continuity

                # Try to scroll down
                await self.page.wait_for_timeout(50)
                scroll_success = await self.page.evaluate(f"""
                    () => {{
                        const prevScrollY = window.scrollY;
                        window.scrollBy(0, {scroll_step});
                        return prevScrollY !== window.scrollY;
                    }}
                """)
                await self.page.wait_for_timeout(100)

                if not scroll_success:
                    # Can't scroll further, end of page reached
                    self.logger.info("Reached end of page, search complete")
                    # Reset for future searches
                    self._last_search_position = 0
                    self._last_search_index = 0
                    self._current_viewport_content = ""
                    self._found_positions = set()
                    return "END OF PAGE REACHED"

                # Get new content after scrolling
                self._current_viewport_content = await self.get_viewport_content()
                self._last_search_position = -1  # Reset for new content

        except Exception as e:
            self.logger.error(f"Error searching for text: {str(e)}")
            # Reset on error
            self._last_search_position = 0
            self._last_search_index = 0
            self._current_viewport_content = ""
            if hasattr(self, "_found_positions"):
                self._found_positions = set()
            return None

    async def get_viewport_content(self) -> str:
        """
        Get the HTML content of the current viewport only.

        Returns:
            str: HTML content in the viewport
        """
        if not self.page:
            self.logger.warning("No page available, can't get viewport content")
            return ""

        # Execute JavaScript to retrieve the text content of elements in viewport
        viewport_content = await self.page.evaluate("""
            () => {
                // Get viewport dimensions
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;
                
                // Function to get text from an element and its children
                function getVisibleText(element, result = []) {
                    // Skip hidden elements and non-content elements
                    if (!element || 
                        !element.getBoundingClientRect ||
                        ['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT'].includes(element.tagName)) {
                        return result;
                    }
                    
                    const rect = element.getBoundingClientRect();
                    
                    // Check if element is at least partially in viewport
                    if (rect.width > 0 && 
                        rect.height > 0 &&
                        rect.bottom > 0 &&
                        rect.right > 0 &&
                        rect.top < viewportHeight &&
                        rect.left < viewportWidth) {
                        
                        // Get this element's text if it has direct text content
                        const ownText = element.childNodes && Array.from(element.childNodes)
                            .filter(n => n.nodeType === 3) // Text nodes only
                            .map(n => n.textContent.trim())
                            .filter(t => t)
                            .join(' ');
                            
                        if (ownText) {
                            result.push(ownText);
                        }
                        
                        // Process child elements
                        if (element.children) {
                            for (const child of element.children) {
                                getVisibleText(child, result);
                            }
                        }
                    }
                    return result;
                }
                
                // Start with body element
                const textContent = getVisibleText(document.body);
                return textContent.join(' ');
            }
        """)

        return viewport_content
