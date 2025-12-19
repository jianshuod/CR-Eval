from openai import OpenAI
from src.analysis.task_worker import Tool
from src.configs import OAI_API_BASE, OAI_API_KEY
from tenacity import retry, stop_after_attempt, wait_random_exponential


gpt_scorer_outline_revision_v5 = Tool("gpt-score-outline-revision-v5", "gpt-4o")
gpt_scorer_diff_analysis_v5 = Tool("gpt-score-diff-analysis-v5", "gpt-4o")
gpt_scorer_fill_cr_v5 = Tool("gpt-score-fill-cr-v5", "gpt-4o")

gpt_client = OpenAI(api_key=OAI_API_KEY, base_url=OAI_API_BASE)


@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(4))
def get_oai_embedding(text):

    if not isinstance(text, str):
        raise ValueError("text should be a string")

    resp = gpt_client.embeddings.create(input=[text], model="text-embedding-3-large")
    embedding = resp.data[0].embedding

    return embedding
