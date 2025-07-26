import streamlit as st
import json
import io
import markdown
from xhtml2pdf import pisa
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# --- Environment and Page Configuration ---

load_dotenv()
st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---


def parse_json_from_string(text: str) -> list:
    """
    Safely extracts a JSON list from a string, even if it's embedded in other text.
    """
    # Finding JSON array part of the string using a regular expression
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from model output. The format was invalid.")
            return []
    st.warning("No valid JSON array found in the model's output.")
    return []


# --- Caching ---


# Cache entire research and report generation process
@st.cache_data(show_spinner=False)
def full_research_process(
    _topic,
    _report_style,
    _num_queries,
    _num_results_per_query,
    _research_depth,
    _use_reranking,
):
    """
    This function encapsulates the entire research and report generation process.
    Streamlit's caching is applied here to avoid re-running the entire process
    for the same inputs.
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    except Exception as e:
        st.error(
            f"Failed to initialize the language model. Check your API key. Error: {e}"
        )
        return None

    parser = StrOutputParser()
    all_sources = []

    # --- Step 1: Generate Initial Search Queries ---
    with st.status(
        "`1. Generating initial search queries...`", expanded=False
    ) as status:
        search_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    "Write {num_queries} google search queries to search online that form an objective opinion from the following: '{question}'\n"
                    "You must respond with a list of strings in the following format: "
                    '["query_1", "query_2", "query_3", ...]',
                ),
            ]
        )

        search_question_chain = search_prompt | model | parser | parse_json_from_string

        try:
            queries = search_question_chain.invoke(
                {"question": _topic, "num_queries": _num_queries}
            )
            if not queries:
                st.error(
                    "Could not generate search queries. The model returned an empty or invalid response. Please try again."
                )
                return None
        except Exception as e:
            st.error(f"An error occurred while generating search queries: {e}")
            return None

        status.update(
            label=f"`1. Generated {len(queries)} initial search queries.`",
            state="complete",
            expanded=False,
        )

    # --- Main Research Loop (with Self-Correction) ---
    for depth in range(_research_depth):
        # --- Step 2: Scrape and Summarize Web Content ---
        with st.status(
            f"`2.{depth + 1} Researching and summarizing sources (Depth {depth + 1}/{_research_depth})...`",
            expanded=True,
        ) as status:
            for i, query in enumerate(queries):
                status.update(
                    label=f"`2.{depth + 1} Researching query: '{query}' ({i + 1}/{len(queries)})...`"
                )

                # Web search - fetch more results for re-ranking
                raw_results = web_search(query, num_results=_num_results_per_query * 2)
                if not raw_results:
                    continue

                # --- Re-ranking Step ---
                if _use_reranking:
                    status.update(
                        label=f"`2.{depth + 1} Prioritizing sources for '{query}'...`"
                    )
                    rerank_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a relevance-ranking assistant. Your task is to reorder a list of URLs based on their relevance to a user's query. Use the provided titles and snippets to make your judgment.",
                            ),
                            (
                                "user",
                                "Based on the query '{query}', please re-rank the following search results from most to least relevant. Respond with only the list of URLs in the new order, in JSON format: ['url1', 'url2', ...]\n\n"
                                "Search Results:\n"
                                "----------------\n"
                                "{results}\n"
                                "----------------\n",
                            ),
                        ]
                    )
                    rerank_chain = (
                        rerank_prompt | model | parser | parse_json_from_string
                    )
                    results_str = "\n".join(
                        [
                            f"URL: {res['link']}\nTitle: {res['title']}\nSnippet: {res['snippet']}"
                            for res in raw_results
                        ]
                    )
                    ranked_links = rerank_chain.invoke(
                        {"query": query, "results": results_str}
                    )
                    links = (
                        ranked_links[:_num_results_per_query]
                        if ranked_links
                        else [
                            res["link"] for res in raw_results[:_num_results_per_query]
                        ]
                    )
                else:
                    links = [
                        res["link"] for res in raw_results[:_num_results_per_query]
                    ]

                # Scrape and summarize each link
                summarize_prompt = ChatPromptTemplate.from_template(
                    """
                    {text}
                    ----------
                    Using the above text, answer in short the following question:
                    > '{question}'
                    ----------
                    If the question cannot be answered using the text, simply summarize the text. Include all factual information, numbers, and stats if available.
                    """
                )
                summarize_chain = (
                    RunnablePassthrough.assign(
                        text=lambda x: scrape_text(x["url"])[:10000]
                    )
                    | summarize_prompt
                    | model
                    | parser
                )

                # Process links in parallel
                summaries = summarize_chain.batch(
                    [{"question": query, "url": link} for link in links]
                )

                # Collect source data
                query_sources = [
                    {"url": link, "summary": summary}
                    for link, summary in zip(links, summaries)
                ]
                all_sources.append({"query": query, "summaries": query_sources})

            status.update(
                label=f"`2.{depth + 1} Complrtrd research and summarization.`",
                state="complete",
                expanded=False,
            )

        # --- Self-Correction Step ---
        if depth < _research_depth - 1:
            with st.status(
                f"`Analyzing research for gaps (Depth {depth + 1})...`", expanded=False
            ) as status:
                research_summary_for_analysis = "\n\n".join(
                    f"Query: {qs['query']}\nSource URL: {s['url']}\nSummary: {s['summary']}"
                    for qs in all_sources
                    for s in qs["summaries"]
                )

                refinement_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a research analyst. Your task is to review the research done so far and identify gaps, unanswered questions, or conflicting viewpoints. Based on this, generate new, specific search queries to deepen the research.",
                        ),
                        (
                            "user",
                            "Original Topic: '{topic}'\n\n"
                            "Research So Far:\n"
                            "----------------\n"
                            "{research_summary}\n"
                            "----------------\n\n"
                            "Based on the research so far, are there any unanswered questions, gaps, or conflicting viewpoints? "
                            "If so, generate a list of {num_queries} new, more specific search queries to address them. "
                            "If the research is already comprehensive, return an empty list.\n"
                            "You must respond with a list of strings in the following format: ['new_query_1', 'new_query_2', ...]",
                        ),
                    ]
                )
                refinement_chain = (
                    refinement_prompt | model | parser | parse_json_from_string
                )
                new_queries = refinement_chain.invoke(
                    {
                        "topic": _topic,
                        "research_summary": research_summary_for_analysis,
                        "num_queries": _num_queries,
                    }
                )

                if new_queries:
                    queries = new_queries
                    status.update(
                        label=f"`Identified {len(queries)} new questions for deeper research.`",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(
                        label="`Analysis complete. No further research required.`",
                        state="complete",
                        expanded=False,
                    )
                    break  # Exit loop if no new queries are generated

    # --- Step 3: Compile Research and Write Final Report ---
    with st.status(
        "`3. Compiling research and writing the final report...`", expanded=False
    ) as status:
        research_summary = "\n\n".join(
            f"Source URL: {s['url']}\nSummary: {s['summary']}"
            for query_source in all_sources
            for s in query_source["summaries"]
        )

        writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI critical thinker research assistant. Your sole purpose is to write well-written, critically acclaimed, objective, and structured reports on a given topic using the provided research material.",
                ),
                (
                    "user",
                    "Information:\n"
                    "----------\n"
                    "{research_summary}\n"
                    "----------\n\n"
                    "Using the above information, write {report_style} on the following topic: '{question}'.\n"
                    "The report should focus on the answer to the question, be well-structured, informative, and in-depth, with facts and numbers if available.\n"
                    "You must write the report with markdown syntax.\n"
                    "You must determine your own concrete and valid opinion based on the given information. Do not deter to general and meaningless conclusions.\n"
                    "Write all used source URLs at the end of the report under a 'References' section. Ensure each URL is on a new line and not duplicated.\n"
                    "You must write the report in APA format.\n"
                    "Please do your best, this is very important to my career.",
                ),
            ]
        )

        report_style_prompts = {
            "Concise Summary": "a concise summary (about 300 words)",
            "Standard Report": "a detailed report (about 800 words)",
            "In-Depth Analysis": "an in-depth, comprehensive analysis (minimum 1500 words), structured with a clear introduction, body, and conclusion. Use headings and subheadings.",
        }

        writer_chain = writer_prompt | model | parser
        final_report = writer_chain.invoke(
            {
                "research_summary": research_summary,
                "question": _topic,
                "report_style": report_style_prompts[_report_style],
            }
        )
        status.update(
            label="`3. Report generation complete!`", state="complete", expanded=False
        )

    return {"report": final_report, "sources": all_sources}


# --- Core Functions ---


def web_search(query: str, num_results: int) -> list[dict]:
    """Performs a web search and returns a list of result dictionaries."""
    search = DuckDuckGoSearchAPIWrapper()
    try:
        results = search.results(query=query, max_results=num_results)
        return results
    except Exception as e:
        st.error(f"DuckDuckGo search failed: {e}")
        return []


def scrape_text(url: str) -> str:
    """Scrapes the text content from a given URL."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as e:
        return f"Failed to retrieve the webpage at {url}: {e}"
    except Exception as e:
        return f"An error occurred while scraping {url}: {e}"


def convert_md_to_pdf(md_content: str) -> bytes:
    """Converts a markdown string to a PDF bytes object."""
    html = markdown.markdown(md_content)
    styled_html = f"""
    <html>
    <head>
    <style>
        @page {{
            size: a4 portrait;
            margin: 2cm;
            @frame header_frame {{
                -pdf-frame-content: header_content;
                left: 1.5cm; right: 1.5cm; top: 1cm; height: 1cm;
            }}
            @frame footer_frame {{
                -pdf-frame-content: footer_content;
                left: 1.5cm; right: 1.5cm; top: 1cm; height: 1cm;
            }}
        }}
        body {{ font-family: "Helvetica", "Arial", sans-serif; font-size: 12pt; line-height: 1.5; color: #333; }}
        h1, h2, h3 {{ font-weight: bold; line-height: 1.2; color: #1a1a1a; }}
        h1 {{ font-size: 24pt; }}
        h2 {{ font-size: 18pt; }}
        h3 {{ font-size: 14pt; }}
        p {{ margin-bottom: 1em; }}
        ul, ol {{ padding-left: 20pt; }}
        a {{ color: #0066cc; text-decoration: none; }}
        pre, code {{ font-family: "Courier New", monospace; background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 10pt; }}
        pre {{ padding: 1em; overflow: auto; }}
        #header_content {{ text-align: left; font-size: 10pt; color: #555; }}
        #footer_content {{ text-align: right; font-size: 10pt; color: #555; }}
    </style>
    </head>
    <body>
        {html}
        <div id="header_content">
            AI Research Report
        </div>
        <div id="footer_content" style="text-align: right; font-size: 10pt; color: #888;">
            Generated by AI Research Assistant | Page <pdf:pageNumber />
        </div>
    </body>
    </html>
    """
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(
        io.StringIO(styled_html), dest=pdf_buffer, encoding="utf-8"
    )
    if pisa_status.err:
        raise Exception(f"Error converting to PDF: {pisa_status.err}")
    return pdf_buffer.getvalue()


# --- Streamlit Application UI ---


def main():
    """Main function to run the Streamlit application."""
    st.title("Advanced AI Research Assistant")
    st.write(
        "This tool uses a multi-step, self-correcting AI agent to conduct research and generate a detailed, objective, and sourced report on any topic. Just enter your question and let the AI do the work!"
    )

    # --- Sidebar for Advanced Controls ---
    with st.sidebar:
        st.header("Research Controls")
        report_style = st.selectbox(
            "Select Report Style",
            ("Concise Summary", "Standard Report", "In-Depth Analysis"),
            index=1,
        )
        num_queries = st.slider(
            "Initial Search Queries", min_value=2, max_value=5, value=3
        )
        num_results_per_query = st.slider(
            "Results per Query", min_value=2, max_value=5, value=3
        )

        st.markdown("---")
        st.header("Advanced Features")
        research_depth = st.slider(
            "Research Depth (Self-Correction Loops)",
            min_value=1,
            max_value=3,
            value=2,
            help="How many times the AI should review its own research and look for gaps. 2 is recommended.",
        )
        use_reranking = st.toggle(
            "Enable Source Re-ranking",
            value=True,
            help="Allow the AI to prioritize the most relevant web search results before summarizing.",
        )
        st.markdown("---")

    # --- Main Content Area ---
    if "result" not in st.session_state:
        st.session_state.result = None

    with st.form("research_form"):
        user_input = st.text_area(
            "Enter your research topic or question:",
            height=100,
            placeholder="e.g., What are the latest advancements in quantum computing?",
        )
        submitted = st.form_submit_button("Generate Report")

    if submitted:
        if user_input.strip():
            st.session_state.result = full_research_process(
                user_input.strip(),
                report_style,
                num_queries,
                num_results_per_query,
                research_depth,
                use_reranking,
            )
        else:
            st.warning("Please enter a research topic.")

    # --- Display Results ---
    if st.session_state.result:
        result = st.session_state.result
        st.markdown("---")
        st.subheader("📄 Generated Report")
        st.markdown(result["report"], unsafe_allow_html=True)

        try:
            pdf_bytes = convert_md_to_pdf(result["report"])
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name="research_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Could not generate PDF for download: {e}")

        with st.expander("View Sources and Summaries"):
            for source_group in result["sources"]:
                st.markdown(f"#### Results for Query: `{source_group['query']}`")
                for item in source_group["summaries"]:
                    st.markdown(f"**URL:** [{item['url']}]({item['url']})")
                    st.markdown(f"**Summary:** {item['summary']}")
                    st.markdown("---")


if __name__ == "__main__":
    main()
