import time
import os
import openai
from .base_language_model import BaseLanguageModel
import dotenv
from pathlib import Path
import importlib.util

repo_root = Path(__file__).resolve().parents[6]
llm_client_path = repo_root / "utils" / "llm_client.py"
llm_client_spec = importlib.util.spec_from_file_location(
    "sageqa_llm_client", llm_client_path
)
if llm_client_spec is None or llm_client_spec.loader is None:
    raise ImportError(f"Could not load SAGE-QA LLM client from {llm_client_path}")
llm_client = importlib.util.module_from_spec(llm_client_spec)
llm_client_spec.loader.exec_module(llm_client)

openai_compatible_client = llm_client.openai_compatible_client

dotenv.load_dotenv()
os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

OPENAI_MODEL = ["gpt-4", "gpt-3.5-turbo"]


def get_token_limit(model="gpt-4"):
    """Returns the token limitation of provided model"""
    if model in ["gpt-4", "gpt-4-0613"]:
        num_tokens_limit = 8192
    elif model in ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]:
        num_tokens_limit = 16384
    elif model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "text-davinci-003",
        "text-davinci-002",
    ]:
        num_tokens_limit = 4096
    else:
        num_tokens_limit = 8192
    return num_tokens_limit


class ChatGPT(BaseLanguageModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--retry", type=int, help="retry time", default=5)

    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150

    def tokenize(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except Exception:
            num_tokens = len(str(text).split())
        return num_tokens + self.redundant_tokens

    def prepare_for_inference(self, model_kwargs={}):
        """
        ChatGPT model does not need to prepare for inference
        """
        pass

    def generate_sentence(self, llm_input):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.tokenize(llm_input)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        while cur_retry <= num_retry:
            try:
                client, model_name, _ = openai_compatible_client(self.model_name)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=query,
                    timeout=30,
                )
                result = response.choices[0].message.content.strip()
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.tokenize(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
