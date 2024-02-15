
import time
from bioimageio_chatbot.utils import ChatbotExtension
from bioimageio_chatbot.chatbot_extensions.web_search_extension.llm_web_search import search_duckduckgo
from bioimageio_chatbot.chatbot_extensions.web_search_extension.langchain_websearch import LangchainCompressor

langchain_compressor = None
async def search_web(query: str):
    """Search the web for information using duckduckgo."""
    global langchain_compressor
    langchain_compressor = langchain_compressor or LangchainCompressor(device="cpu")
    content = await search_duckduckgo(query, langchain_compressor, max_results=5, similarity_threshold=0.5, instant_answers=True, chunk_size=500, num_results_to_process=5)
    return content

def get_extensions():
    
    return [
        ChatbotExtension(
            name="SearchWeb",
            description="Search the web for information using duckduckgo.",
            execute=search_web,
        )
    ]

# if __name__ == "__main__":
#     langchain_compressor = LangchainCompressor(device="cpu")
#     start = time.time()
#     content = await search_duckduckgo("main contributor deepimagej", langchain_compressor, max_results=5, similarity_threshold=0.5, instant_answers=True, chunk_size=500, num_results_to_process=5)
#     end = time.time()
#     print(content)
#     print(f"Time: {end - start}")

