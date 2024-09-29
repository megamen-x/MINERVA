#openai_embeddings_llm.py

from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

from FlagEmbedding import BGEM3FlagModel

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(
            self,
            client: OpenAIClientTypes,
            configuration: OpenAIConfiguration
    ):
        self._client = client
        self._configuration = configuration

        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True
        )

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embeddings_1 = self.model.encode(
            input,
            batch_size=1,
            max_length=8192
        )['dense_vecs']

        embeddings_list = [embs.tolist() for embs in embeddings_1]
        return embeddings_list
