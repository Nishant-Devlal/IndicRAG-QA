import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="IndicRAG-QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 16px;
            background: linear-gradient(
                120deg,
                #172554,
                #1d4ed8
            );
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.4rem;
        }

        .hero p {
            color: #dbeafe;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .status {
            padding: 0.7rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.9rem;
        }

        .status-success {
            background-color: #dcfce7;
            color: #166534;
        }

        .status-warning {
            background-color: #fef3c7;
            color: #92400e;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.85rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 12px;
        }

        .stChatInput textarea {
            border-radius: 12px;
        }

        .scheme-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            grid-auto-rows: 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .scheme-card {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            background: rgba(128, 128, 128, 0.06);
            box-sizing: border-box;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }

        .scheme-card:hover {
            transform: translateY(-3px);
            border-color: #3b82f6;
        }

        .scheme-card h3 {
            margin: 0 0 0.6rem 0;
            font-size: 1.15rem;
            line-height: 1.3;
        }

        .scheme-card p {
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.5;
            opacity: 0.9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline(tfidf_weight=0.9, dense_weight=0.1, candidate_k=10, top_k=5,)

with st.sidebar:
    st.title("📚 IndicRAG-QA")
    st.caption("Evidence-Grounded Cross-Lingual Question Answering")
    st.divider()
    st.subheader("About the Project")
    st.write(
        "IndicRAG-QA is a domain-specific RAG chatbot "
        "designed to answer questions about government "
        "education schemes using retrieved document evidence."
    )
    st.subheader("Supported Languages")
    st.markdown(
        """
        - English
        - Hindi
        - Hinglish
        """
    )
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <h1>IndicRAG-QA 📚</h1>
        <p>
            Ask questions about government education schemes
            in English, Hindi, or Hinglish.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Supported Government Schemes")
st.markdown("Explore the government education schemes currently covered by IndicRAG-QA.")

schemes = [
    {
        "name": "PM POSHAN",
        "description": "Provides mid-day meals to school children.",
    },
    {
        "name": "PM SHRI",
        "description": "Develops model schools that showcase the National Education Policy (NEP) 2020 through modern and quality education.",
    },
    {
        "name": "RTE (Right to Education)",
        "description": "Guarantees free and compulsory education for all children aged 6 to 14 years.",
    },
    {
        "name": "Samagra Shiksha",
        "description": "An integrated school education programme covering different stages of school education.",
    },
    {
        "name": "PM-USP Scholarship",
        "description": "A scholarship initiative supporting eligible students pursuing higher education.",
    },
    {
        "name": "National Means-cum-Merit Scholarship",
        "description": "A scholarship scheme intended to support eligible students from economically weaker sections.",
    },
    {
        "name": "Prime Minister's Scholarship Scheme",
        "description": "A scholarship scheme for eligible students under its applicable guidelines.",
    },
    {
        "name": "PM YASASVI",
        "description": "Provides scholarships and educational support to OBC, EBC, and DNT students for quality education.",
    },
    {
        "name": "PM-USHA",
        "description": "Supports the improvement of quality, accessibility, and equity in higher education institutions.",
    },
    {
        "name": "PM Vidyalaxmi Scheme",
        "description": "Facilitates easy access to education loans for students pursuing higher education.",
    },
    {
        "name": "RUSA (Rashtriya Uchchatar Shiksha Abhiyan)",
        "description": "Aims to improve the quality and governance of state higher education institutions.",
    },
    {
        "name": "SHREYAS",
        "description": "Provides scholarships and skill development opportunities for Scheduled Caste students in higher education.",
    },
    {
        "name": "KGBV (Kasturba Gandhi Balika Vidyalaya)",
        "description": "Establishes residential schools for girls from disadvantaged and marginalized communities.",
    },
    {
        "name": "BBBP (Beti Bachao Beti Padhao)",
        "description": "Promotes the education, empowerment, and welfare of the girl child while addressing gender imbalance.",
    },
]

cards_html = "".join(
    f'<div class="scheme-card"><h3>{scheme["name"]}</h3><p>{scheme["description"]}</p></div>'
    for scheme in schemes
)
st.markdown(f'<div class="scheme-grid">{cards_html}</div>', unsafe_allow_html=True)

st.info("More government schemes and documents will be added soon!")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            answerable = message.get("answerable")

            if answerable is True:
                st.markdown(
                    """
                    <div class="status status-success">
                        ✓ Answerability check passed
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif answerable is False:
                st.markdown(
                    """
                    <div class="status status-warning">
                        ! No sufficient supporting evidence detected
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            sources = message.get("sources", [])

            if sources:
                with st.expander(f"📄 View Sources ({len(sources)})"):
                    for source in sources:
                        st.markdown(
                            f"""
                            **Chunk ID:** {source.get('chunk_id', 'N/A')}

                            **Document:** {source.get('document_id', 'N/A')}

                            **Language:** {source.get('language', 'N/A')}

                            **Page:** {source.get('page', 'N/A')}
                            """
                        )
                        st.divider()

pending_question = st.session_state.pop("pending_question", None,)
user_question = st.chat_input("Ask a question about government education schemes...")
question = pending_question or user_question

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user", avatar="🧑🏼‍🦰"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Retrieving evidence and preparing your answer..."):
                pipeline = load_pipeline()
                result = pipeline.answer(question)

            answer = result.get("answer", "No answer was returned by the pipeline.",)
            answerable = result.get("answerable")
            sources = result.get("sources", [])

            st.markdown(answer)

            if answerable is True:
                st.markdown(
                    """
                    <div class="status status-success">
                        ✓ Answerability check passed
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif answerable is False:
                st.markdown(
                    """
                    <div class="status status-warning">
                        ! No sufficient supporting evidence detected
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if sources:
                with st.expander(f"📄 View Sources ({len(sources)})", expanded=False,):
                    for source in sources:
                        st.markdown(
                            f"""
                            **Chunk ID:** {source.get('chunk_id', 'N/A')}
                            **Document:** {source.get('document_id', 'N/A')}
                            **Language:** {source.get('language', 'N/A')}
                            **Page:** {source.get('page', 'N/A')}
                            """
                        )
                        st.divider()

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "answerable": answerable,
                "sources": sources,
            }

            st.session_state.messages.append(assistant_message)

        except Exception as error:
            error_message = ("Something went wrong while processing your question. "
                             "Please check the terminal for the error details.")

            st.error(error_message)

            print("IndicRAG-QA error:", repr(error))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "answerable": False,
                    "sources": [],
                }
            )