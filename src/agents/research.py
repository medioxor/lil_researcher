import datetime
import logging
from typing import Dict, List
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from logging import Logger
from .browser import BrowserAgent
from serpapi import GoogleSearch
import os


class ResearchAgent:
    def __init__(
        self,
        model_url="http://ollama:11434/v1",
        model_name="custom-model",
        max_queries_per_question=3,
        max_questions_per_url=3,
        max_urls_per_question=20,
    ):
        self.generic_system_prompt = (
            f"Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}. You will use the most up-to-date information possible.",
        )
        self.max_queries_per_question = max_queries_per_question
        self.max_questions_per_url = max_questions_per_url
        self.max_urls_per_question = max_urls_per_question
        self.ollama_model = OpenAIModel(model_name=model_name, base_url=model_url)
        self.research_agent = Agent(
            self.ollama_model,
            result_type=List[str],
            retries=20,
            system_prompt=(
                "You are an expert in completing tasks that require researching questions or topics.",
                'Any questions you generate must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.',
            )
            + self.generic_system_prompt,
        )
        self.query_agent = Agent(
            self.ollama_model,
            result_type=List[str],
            retries=20,
            system_prompt=(
                "You are an expert at generating search engine queries to better undertand a question or topic.",
                "Create as many questions as you can to explore a given question or topic using search engine queries.",
                "Ensure that each query is unique and relevant to the question or topic.",
                'You will not prepend the queries with "-", "1." or any other special characters.',
            )
            + self.generic_system_prompt,
        )
        self.summarize_agent = Agent(
            self.ollama_model,
            result_type=str,
            retries=20,
            system_prompt=(
                "You are an expert at contextualizing information.",
                "Your goal is to provide the most accurate information.",
                "Do not hallucinate or make up information.",
                "Do not include tool calls in your response.",
                "Do not include the question in your response.",
                "Do not include your thinking process in your response.",
            )
            + self.generic_system_prompt,
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
            if len(result) == self.max_queries_per_question:
                return result
            else:
                raise ModelRetry(
                    f"You generated {len(result)} queries, you needed to generate {self.max_queries_per_question}, try again"
                )

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
            if len(result) == self.max_questions_per_url:
                return result
            else:
                raise ModelRetry(
                    f"You generated {len(result)} questions, you needed to generate {self.max_questions_per_url}, try again"
                )

    async def search_google(self, query: str) -> List[str]:
        """Search google for URLs based on a search query.

        Args:
            query: The query to use when searching google

        Returns:
            List[str]: A list of URLs based on the search query
        """
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            self.logger.error(
                "SERPAPI_API_KEY environment variable not set, bailing out"
            )
            return []

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        try:
            if "organic_results" not in results:
                self.logger.warning("No organic_results in search results")
                return []

            organic_results = results["organic_results"]
            return [str(result["link"]) for result in organic_results]
        except Exception as e:
            self.logger.error(f"Error processing search results: {e}")
            return []

    async def research(self, input: str) -> str:
        self.logger.info("=" * 80)
        self.logger.info(f"📋 RESEARCH TASK: {input}")
        self.logger.info("=" * 80)

        # Generate search queries
        self.logger.info("🔍 GENERATING SEARCH QUERIES...")
        result = await self.query_agent.run(
            f"""
            Generate 3 of your best google search queries that you can use to research how to complete the following task:
            {input}
            """
        )
        for i, query in enumerate(result.data, 1):
            self.logger.info(f"  Query #{i}: {query}")
        queries = result.data
        answers: Dict[str, List[str]] = {}
        urls = []

        # Search for URLs
        self.logger.info("\n📊 SEARCHING GOOGLE FOR RELEVANT URLS...")
        for i, query in enumerate(queries[: self.max_queries_per_question], 1):
            self.logger.info(f"  Searching query #{i}: {query}")
            search_results = await self.search_google(query)
            self.logger.info(f"    Found {len(search_results)} URLs")
            urls.extend(search_results)

        urls = list(set(urls))
        self.logger.info(f"\n🌐 TOTAL UNIQUE URLS FOUND: {len(urls)}")
        for i, url in enumerate(urls, 1):
            self.logger.info(f"  URL #{i}: {url}")

        # Generate research questions
        self.logger.info("\n❓ GENERATING RESEARCH QUESTIONS...")
        questions = await self.research_agent.run(
            f"""
            Return a list of questions that you can use to research "{input}", ensure you pick the top 3 best questions.
            """
        )
        for i, question in enumerate(questions.data, 1):
            self.logger.info(f"  Question #{i}: {question}")

        # Process URLs and questions
        self.logger.info("\n📚 ANALYZING URLS WITH QUESTIONS...")
        for i, url in enumerate(urls, 1):
            self.logger.info(f"\n  Analyzing URL #{i}: {url}")
            answers[url] = []
            for j, question in enumerate(
                questions.data[: self.max_questions_per_url], 1
            ):
                self.logger.info(f"    Question #{j}: {question}")
                try:
                    answer = await self.browser_agent.ask_question(url, question)
                    if answer:
                        self.logger.info(
                            f"      ✅ Answer received ({len(answer)} chars)"
                        )
                        answers[url].append(answer)
                    else:
                        self.logger.warning("      ⚠️ No answer received")
                except Exception as e:
                    self.logger.error(f"      ❌ Error analyzing URL: {e}")
                    break

        # Generate final answer
        self.logger.info("\n📝 GENERATING FINAL ANSWER...")
        final_answer = ""

        for i, url in enumerate(answers.keys(), 1):
            self.logger.info(f"  Processing URL #{i}: {url}")
            self.logger.info(f"    Using {len(answers[url])} answers from this URL")

            result = await self.summarize_agent.run(
                f"""
                It is of utmost importance that you improve the final answer in relation to the question "{input}" with the information provided and add references.
                Use the following answers based on the "{url}" URL to improve the final answer which should answer {input}: 
                    {"\t\n".join(answers[url])}
                Do not return anything but the improved final answer, the current final answer is as follows which starts with "START FINAL ANSWER" and ends with "END FINAL ANSWER":
                START FINAL ANSWER
                {final_answer}
                END FINAL ANSWER
                The answer you return should be a research paper quality answer in markdown format.
                """
            )
            final_answer = result.data
            self.logger.info(f"    Answer updated (now {len(final_answer)} chars)")

        self.logger.info("\n✅ RESEARCH COMPLETED")
        self.logger.info("=" * 80)
        return final_answer
