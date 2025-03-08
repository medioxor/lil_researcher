import logging
from typing import Dict, List
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from logging import Logger
from .browser import BrowserAgent
from duckduckgo_search import DDGS


class ResearchAgent:
    def __init__(
        self,
        model_url="http://ollama:11434/v1",
        model_name="custom-model",
    ):
        self.ollama_model = OpenAIModel(model_name=model_name, base_url=model_url)
        self.research_agent = Agent(
            self.ollama_model,
            result_type=List[str],
            retries=20,
            system_prompt=(
                "You are an expert in researching requests, questions or topics.",
                'Your questions must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.',
            ),
        )
        self.query_agent = Agent(
            self.ollama_model,
            result_type=List[str],
            retries=20,
            system_prompt=(
                "You are an expert at generating search engine queries to better undertand a question or topic.",
                "Create as many questions as you can to explore a given question or topic using search engine queries.",
            ),
        )
        self.summarize_agent = Agent(
            self.ollama_model,
            result_type=str,
            retries=20,
            system_prompt=(
                "You are an expert at contextualizing information.",
                "Your goal is to provide the most accurate information based on the context of the information you are summarizing.",
                "Do not remove the context of the information you are summarizing.",
                "Do not hallucinate or make up information.",
                "Do not include tool calls in your response.",
                "Do not include the question in your response.",
                "Do not include your thinking process in your response.",
            ),
        )
        self.browser_agent = BrowserAgent()
        self.logger: Logger = logging.getLogger("research_agent")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._register_validators()

    def _register_validators(self):
        @self.query_agent.result_validator
        async def validate_queries(result: List[str]) -> List[str]:
            if len(result) > 0:
                return result
            else:
                raise ModelRetry("No queries generated, try again")

        @self.summarize_agent.result_validator
        async def validate_summary(result: str) -> str:
            if result.startswith("```json"):
                raise ModelRetry(
                    "Invalid summary, do not include tool calls in your response"
                )
            if len(result) > 0:
                return result
            else:
                raise ModelRetry("Invalid summary, try again")

        @self.research_agent.result_validator
        async def validate_questions(result: str) -> str:
            if len(result) > 0:
                return result
            else:
                raise ModelRetry("No questions generated, try again")

    async def search_google(self, query: str) -> List[str]:
        """Search google for URLs based on a search query.

        Args:
            query: The query to use when searching google

        Returns:
            List[str]: A list of URLs based on the search query
        """
        try:
            results = DDGS().text(query, max_results=20, safesearch="off")
            self.logger.info(f"Search results: {len(results) if results else 0}")
            urls = [
                result["href"]
                for result in results
                if isinstance(result, dict) and "href" in result
            ]
            return urls
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    async def research(self, input: str) -> str:
        result = await self.query_agent.run(
            f"""
            Return a list of google search queries that you can use to research the input based on the following question/topic:
            {input}
            """
        )

        queries = result.data
        answers: Dict[str, List[str]] = {}
        urls = []

        for query in queries:
            search_results = await self.search_google(query)
            urls.extend(search_results)

        urls = list(set(urls))
        questions = await self.research_agent.run(
            f"""
            Return a list of questions that you can use to research "{input}"
            """
        )

        for url in urls:
            answers[url] = []
            for question in questions.data:
                try:
                    answer = await self.browser_agent.ask_question(url, question)
                except Exception as e:
                    self.logger.error(f"Error analyzing URL: {e}")
                    break
                if answer:
                    answers[url].append(answer)

        final_answer = ""

        for url in answers.keys():
            self.logger.info(f"Current final answer: {final_answer}")
            result = await self.summarize_agent.run(
                f"""
                It is of utmost importance that you improve the final answer in relation to the question "{input}" with the information provided and add references.
                Use the following answers based on the "{url}" URL to improve the final answer which should answer {input}: 
                    {"\t\n".join(answers[url])}
                Do not return anything but the final answer, the final answer is as follows which starts with "START FINAL ANSWER" and ends with "END FINAL ANSWER":
                START FINAL ANSWER
                {final_answer}
                END FINAL ANSWER
                """
            )
            self.logger.info(f"Improved final answer: {result.data}")
            final_answer = result.data

        return final_answer
