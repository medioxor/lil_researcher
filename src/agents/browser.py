from ..tools.browser import Browser
from pydantic_ai.models.openai import OpenAIModel
from dataclasses import dataclass
from pydantic_ai import Agent, ModelRetry, RunContext
import logging


@dataclass
class BrowserDeps:
    browser: Browser


class BrowserAgent:
    def __init__(
        self,
        model_url="http://ollama:11434/v1",
        model_name="custom-model",
    ):
        self.ollama_model = OpenAIModel(model_name=model_name, base_url=model_url)
        self.agent = Agent(
            self.ollama_model,
            result_type=str,
            retries=20,
            system_prompt=(
                "Ensure you read all contents on the page.",
                "Ensure you break up the question into smaller parts and search for each part.",
                "When using `page_up` or `page_down`, if you receive 'END OF PAGE REACHED' or 'TOP OF PAGE REACHED', you must stop.",
                "When using `find_text` or `find_text_next`, ensure the text you provide is not entire sentences, but rather keywords.",
            ),
        )
        self.logger: logging.Logger = logging.getLogger("browser_agent")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._register_tools()
        self._register_validators()

    def _register_validators(self):
        @self.agent.result_validator
        async def validate_answer(result: str) -> str:
            if result.startswith("```json"):
                raise ModelRetry(
                    "Invalid answer, do not include tool calls in your response"
                )
            if len(result) > 0:
                return result
            else:
                raise ModelRetry(
                    "No answer provided, try again but ensure you find the answer or provide context on why you couldn't find it."
                )

    def _register_tools(self):
        @self.agent.tool(docstring_format="google", require_parameter_descriptions=True)
        async def page_up(ctx: RunContext[BrowserDeps]) -> str:
            """Scroll up to the previous viewport, showing completely new content.

            Args:
                ctx: The run context containing dependencies.

            Returns:
                str: Content of the viewport after scrolling, "TOP OF PAGE REACHED" if at top,
                or empty string if error occurred.
            """
            self.logger.info("Calling page_up")
            result = await ctx.deps.browser.page_up()
            return result

        @self.agent.tool(docstring_format="google", require_parameter_descriptions=True)
        async def page_down(ctx: RunContext[BrowserDeps]) -> str:
            """Scroll down to the next viewport, showing completely new content.

            Args:
                ctx: The run context containing dependencies.

            Returns:
                str: Content of the viewport after scrolling, "END OF PAGE REACHED" if at bottom,
                or empty string if error occurred.
            """
            self.logger.info("Calling page_down")
            result = await ctx.deps.browser.page_down()
            return result

        @self.agent.tool(docstring_format="google", require_parameter_descriptions=True)
        async def find_text(ctx: RunContext[BrowserDeps], search_text: str) -> str:
            """Search for text in the page, scrolling through it until found or end is reached.

            Args:
                ctx: The run context containing dependencies.
                search_text: The text to search for in the page.

            Returns:
                str: Viewport content containing the text, or None if not found.
            """
            self.logger.info(f"Calling find_text with search_text: {search_text}")
            result = await ctx.deps.browser.find_text(search_text)
            self.logger.info(
                f"find_text returning content of length: {len(result) if result else 0}"
            )
            return result

        @self.agent.tool(docstring_format="google", require_parameter_descriptions=True)
        async def find_text_next(ctx: RunContext[BrowserDeps], search_text: str) -> str:
            """Search for the next occurrence of text in the page.

            Args:
                ctx: The run context containing dependencies.
                search_text: The text to search for in the page.

            Returns:
                str: Viewport content containing the text, or None/END OF PAGE REACHED if not found.
            """
            self.logger.info(f"Calling find_text_next with search_text: {search_text}")
            result = await ctx.deps.browser.find_text_next(search_text)
            self.logger.info(
                f"find_text_next returning content of length: {len(result) if result else 0}"
            )
            return result

        @self.agent.tool(docstring_format="google", require_parameter_descriptions=True)
        async def get_viewport_content(ctx: RunContext[BrowserDeps]) -> str:
            """Get the HTML content of the current viewport only.

            Args:
                ctx: The run context containing dependencies.

            Returns:
                str: HTML content in the viewport.
            """
            self.logger.info("Calling get_viewport_content")
            result = await ctx.deps.browser.get_viewport_content()
            self.logger.info(
                f"get_viewport_content returning content of length: {len(result) if result else 0}"
            )
            return result

    async def ask_question(self, url: str, question: str) -> str:
        browser = Browser()
        await browser.start()
        if not await browser.navigate(url):
            raise Exception(f"Failed to navigate to URL: {url}")

        deps = BrowserDeps(browser=browser)
        result = await self.agent.run(
            f"""
            You have one question to answer. It is paramount that you provide a correct answer.
            Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
            Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
            Run verification steps if that's needed, you must make sure you find the correct answer!
            Here is the task:
            {question}  
            """,
            deps=deps,
        )
        await browser.close()
        return result.data
