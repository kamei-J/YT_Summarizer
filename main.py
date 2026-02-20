from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
import re
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(override=True)

llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.8)

class GraphState(BaseModel):
    video_url: str = Field(description="The URL of the youtube video")
    video_id: Optional[str] = Field(default=None, description="The ID of the youtube video")
    transcript: Optional[str] = Field(default=None, description="The transcript of the youtube video")
    summary: Optional[str] = Field(default=None, description="The summary of the youtube video transcript")
    keyword: Optional[str] = Field(default=None, description="The key word extracted from the youtube video transcript")
    video_suggestions: Optional[list[str]] = Field(default=None, description="The suggested title and description for the youtube video")
    questions: Optional[str] = Field(default=None, description="The suggested questions based on the youtube video transcript")
    next_steps: Optional[str] = Field(default=None, description="The suggested next steps based on the summary of the youtube video transcript")

class ExtractedVideoID(BaseModel):
    video_id: str = Field(description="The ID of the youtube video")

def extract_video_id(state: GraphState):
    video_url = state.video_url
    template = PromptTemplate(
        template='''
        Extract the video ID from the following YouTube URL: {video_url}
        Return only the video ID.
        ''',
        input_variables=["video_url"]
    )
    llm_with_structured_output = llm.with_structured_output(ExtractedVideoID)
    chain = template | llm_with_structured_output
    response = chain.invoke({"video_url": video_url})
    return {"video_id": response.video_id}

def extract_transcript(state: GraphState):
    video_id = state.video_id
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_text = " "
    for snippet in fetched_transcript:
        transcript_text += " " + snippet.text
    return {"transcript": transcript_text}

def summarize_transcript(state: GraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template='''
        Summarize the following transcript in a concise manner:
        {transcript}
        ''',
        input_variables=["transcript"]
    )
    chain = template | llm
    response = chain.invoke({"transcript": transcript})
    return {"summary": response.content}

def generate_questions(state: GraphState):
    summary = state.summary
    template = PromptTemplate(
        template='''
        Generate 5 questions based on the following summary:
        {summary}
        ''',
        input_variables=["summary"]
    )
    chain = template | llm
    response = chain.invoke({"summary": summary})
    return {"questions": response.content}

def next_steps(state: GraphState):
    summary = state.summary
    template = PromptTemplate(
        template='''
        Based on the following summary, suggest the next steps:
        {summary}
        For example, if the video is about React basic, suggest learning about state management or hooks.
        ''',
        input_variables=["summary"]
    )
    chain = template | llm
    response = chain.invoke({"summary": summary})
    return {"next_steps": response.content}

def find_keywords(state: GraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template='''
        Extract the most relevant keyword from the following transcript:
        {transcript}
        The keyword should be a single word or a short phrase that best represents the main topic of the transcript.
        For example, if the video is about React basic, return "React".
        Return only the keyword. 
        ''',
        input_variables=["transcript"]
    )
    chain = template | llm
    response = chain.invoke({"transcript": transcript})
    return {"keyword": response.content}

def video_suggestion(state: GraphState):
    keyword = state.keyword
    if not keyword:
        return {"video_suggestions": []}

    def _clean_keyword(k: str) -> str:
        s = str(k or "").strip()
        # prefer quoted content
        m = re.search(r'"([^"]+)"', s)
        if m:
            return m.group(1).strip()
        m = re.search(r"'([^']+)'", s)
        if m:
            return m.group(1).strip()
        # drop leading labels like 'the most relevant keyword is:'
        s = re.sub(r'^(.*?:)\s*', '', s).strip()
        # take the last non-empty line
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        # remove surrounding quotes/punctuation
        s = s.strip(' "\' .')
        # remove common prefix text
        s = re.sub(r'^(the most relevant keyword is[:\s-]*)', '', s, flags=re.IGNORECASE).strip()
        return s

    query = _clean_keyword(keyword)
    if not query:
        return {"video_suggestions": []}

    tool = YouTubeSearchTool()
    # YouTubeSearchTool expects a query optionally followed by a result count.
    # Provide an explicit count to avoid parsing issues if the LLM output contains extra text.
    try:
        video_suggestions = tool.run(f"{query}||5")
    except Exception:
        # fallback to calling with just the cleaned query
        video_suggestions = tool.run(query)
    return {"video_suggestions": video_suggestions}

app = StateGraph(GraphState)

app.add_node("extract_video_id", extract_video_id)
app.add_node("extract_transcript", extract_transcript)
app.add_node("summarize_transcript", summarize_transcript)
app.add_node("generate_questions", generate_questions)
app.add_node("next_steps", next_steps)
app.add_node("find_keywords", find_keywords)
app.add_node("video_suggestion", video_suggestion)

app.add_edge(START, "extract_video_id")
app.add_edge("extract_video_id", "extract_transcript")
app.add_edge("extract_transcript", "summarize_transcript")
app.add_edge("summarize_transcript", "generate_questions")
app.add_edge("summarize_transcript", "next_steps")
app.add_edge("extract_transcript", "find_keywords")
app.add_edge("find_keywords", "video_suggestion")
app.add_edge("generate_questions", END)
app.add_edge("next_steps", END)
app.add_edge("video_suggestion", END)

graph = app.compile()