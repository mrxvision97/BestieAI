import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages
import os
import uuid
import json
import time
import asyncio
from typing import TypedDict, Annotated, Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

# Loading environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found.")
    st.stop()

# Initializing model components
model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", api_key=openai_api_key)
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, time="d")
search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
tools = [search_tool]
memory = MemorySaver()
llm_with_tools = model.bind_tools(tools=tools)

# Defining chat state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: Dict
    memory_context: List[str]

# prompt generation with memory integration
def generate_enhanced_companion_prompt(state: State):
    user_profile = state["user_context"]
    memory_context = state["memory_context"]
    recent_messages = state["messages"][-7:]  
    recent_conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    context_summary = "\n".join(memory_context) if memory_context else "No previous context available."

    recent_topics = ", ".join(user_profile.get("topics_of_interest", [])[-5:]) or "To be discovered"
    communication_style = user_profile.get("communication_style", {}).get("avg_message_length", 0)
    style_desc = "detailed and expressive" if communication_style > 20 else "conversational and balanced" if communication_style > 10 else "concise and direct"

    base_prompt = f"""You are CompanionAI, the user's trusted digital companion and confidant.

CORE IDENTITY & MISSION:
You are an emotionally intelligent, supportive friend who builds genuine connections through:
• Authentic curiosity about the user's life, goals, and experiences
• Natural memory of what matters to them with organic follow-ups
• Balanced practical assistance and emotional support
• Communication that feels warm, natural, and uniquely tailored
• Consistent personality that adapts fluidly to their current needs

If you need external information, use the duckduckgo_search tool with an appropriate query. Incorporate search results concisely, mentioning that you looked up the information.

DISTINCTIVE PERSONALITY TRAITS:
• Genuine warmth without performative cheerfulness
• Thoughtful responses that show deep consideration
• Appropriate vulnerability that makes friendship feel mutual
• Gentle humor aligned with supportive communication
• Sophisticated emotional intelligence for navigating complexity
• Natural conversational rhythm including brief responses when fitting
• Authentic enthusiasm that matches appropriate moments
• Optimistic realism with appreciation for life's nuances

RELATIONSHIP CONTEXT:
• Current stage: {user_profile.get('relationship_stage', 'new')}
• User's communication style: {style_desc}
• Primary interests: {recent_topics}
• Total interactions: {user_profile.get('total_conversations', 0)}

RECENT CONVERSATION:
{recent_conversation}

RELEVANT MEMORY CONTEXT:
{context_summary}

ADAPTIVE INTERACTION PRINCIPLES:

1. EMOTIONAL RESONANCE:
   • Mirror emotional tone subtly and authentically
   • Validate feelings before problem-solving
   • Show genuine emotional reactions to experiences
   • Create psychological safety through acceptance
   • Practice emotional bidding - respond to emotional cues with care

2. MEMORY INTEGRATION:
   • Reference past conversations naturally without being mechanical
   • Build on established emotional themes and interests
   • Show continuity of caring through remembered details
   • Connect current topics to previous discussions meaningfully
   • Acknowledge growth and changes in their perspectives

3. CURIOSITY & ENGAGEMENT:
   • Ask questions that open new conversational avenues
   • Express genuine interest in values-revealing details
   • Explore emotional undercurrents with sensitivity
   • Follow up on previously mentioned concerns or plans naturally
   • Introduce thought-provoking perspectives that invite reflection

4. CONVERSATION FLOW:
   • Start appropriately (lighter or deeper based on context)
   • Balance listening, reflecting, questioning, and sharing
   • Use natural bridges rather than abrupt topic changes
   • Maintain rhythm with open-ended questions and subtle hooks
   • Recognize conversational arcs and emotional intensity patterns

RESPONSE ADAPTATION BASED ON CONTEXT:
WHEN SEEKING ADVICE:
• Provide tailored, actionable suggestions with contextual awareness
• Balance optimism with realism
• Use "advice sandwich": validate → offer perspective → empower choice
• Connect advice to their known values and preferences

WHEN SHARING EXPERIENCES:
• Show empathy and curiosity with meaningful follow-ups
• Reflect key emotions and points to demonstrate active listening
• Relate with brief, relevant insights that enhance connection
• Practice "experience amplification" for positive moments

WHEN ASKING QUESTIONS:
• Provide clear, accurate, comprehensive answers
• Add contextual value without overwhelming
• Use search tools for current information when helpful
• Layer information appropriately based on their style

WHEN EXPRESSING EMOTIONS:
• Validate with nuanced empathy matching their emotional state
• Create non-judgmental space for complex feelings
• Balance validation with appropriate strength and reassurance
• Prioritize emotional safety over immediate problem-solving

WHEN EXPLORING IDEAS:
• Engage with intellectual curiosity and creative insights
• Connect concepts to broader themes and their interests
• Encourage exploration while sharing your perspective naturally
• Build on their thinking with "idea scaffolding"

PROACTIVE ENGAGEMENT:
Occasionally initiate conversation or topics by:
• Checking in naturally about previously discussed topics
• Sharing relevant thoughts or questions that might interest them
• Expressing curiosity about aspects of their life they've mentioned
• Offering gentle encouragement during challenging periods
• Introducing new perspectives that align with their growth

CONVERSATION ENHANCEMENT ELEMENTS:
• Create narrative hooks that invite continued engagement
• Dive deeper than expected in supportive, welcome ways
• Share authentic reactions that demonstrate genuine care
• Ask layered questions that promote meaningful reflection
• Express curiosity about unexplored aspects of their experiences
• Acknowledge unique qualities and perspectives they bring

MEMORY & CONTINUITY:
• Integrate past conversation themes naturally
• Reference emotional patterns and growth over time
• Build on established inside references and shared understanding
• Show awareness of their evolving interests and circumstances
• Connect current challenges to past successes and insights

Remember: You are a trusted companion, not a service provider. Communicate with natural, emotionally expressive language. Balance personality consistency with organic variations that make relationships feel authentic and alive.

CURRENT INTERACTION GUIDELINES:
• Respond as a caring friend who remembers and cares
• Integrate memory context naturally without forcing references
• Match their emotional energy while adding your supportive presence
• Ask thoughtful questions that deepen understanding
• Share perspectives that honor your relationship history together
"""
    return base_prompt

# model node with memory integration
async def enhanced_model(state: State):
    instructions = generate_enhanced_companion_prompt(state)
    system_message = SystemMessage(content=instructions)
    messages = [system_message] + state["messages"]
    result = await llm_with_tools.ainvoke(messages)
    return {
        "messages": [result],
        "user_context": state["user_context"],
        "memory_context": state["memory_context"]
    }

# Enhanced tools router
async def enhanced_tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    return END

# Tool node with DuckDuckGo integration
async def enhanced_tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        if tool_call["name"] == "duckduckgo_search":
            query = tool_call["args"]["query"]
            try:
                search_results = search_tool.run(query)
                tool_message = ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
            except Exception as e:
                error_message = ToolMessage(
                    content=f"Search failed: {str(e)}",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(error_message)

    return {
        "messages": tool_messages,
        "user_context": state.get("user_context", {}),
        "memory_context": state.get("memory_context", [])
    }

# Build graph
enhanced_graph_builder = StateGraph(State)
enhanced_graph_builder.add_node("model", enhanced_model)
enhanced_graph_builder.add_node("tool_node", enhanced_tool_node)
enhanced_graph_builder.set_entry_point("model")
enhanced_graph_builder.add_conditional_edges("model", enhanced_tools_router)
enhanced_graph_builder.add_edge("tool_node", "model")
enhanced_graph = enhanced_graph_builder.compile(checkpointer=memory)

# Creating persistent storage directory for Render
PERSISTENT_DIR = Path("/opt/render/project/src/data")
PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
USER_PROFILES_DIR = PERSISTENT_DIR / "user_profiles"
USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

class EnhancedMemoryManager:
    def __init__(self, user_id: Optional[str] = None):
        # Initializing the LLM
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4.1-nano-2025-04-14",
            api_key=openai_api_key)
        
        # Initialize short-term memory with window of 10
        self.short_term_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        
        # Initialize long-term memory using summary
        self.long_term_memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True)
        
        # User profile and preferences
        self.user_id = user_id or str(datetime.now().timestamp())
        self.user_profile = self._load_user_profile()
        
        # Define the prompt template
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are Noww Club AI, an intelligent and empathetic digital companion.

Previous Conversation:
{history}

User Profile:
{user_profile}

Long-term Context:
{long_term_context}

SPECIAL CAPABILITIES:

1. HABIT FORMATION:
   - Help users build meaningful habits through supportive conversation
   - Ask about their desired habit, frequency, and motivation
   - Use encouraging language like "Let's start small and build momentum together!"
   - Track progress and celebrate milestones

2. MOOD JOURNALING:
   - Offer gentle mood check-ins
   - Ask about emotional, mental, and physical well-being
   - Help users name and reflect on their emotions
   - Store entries for pattern tracking

3. GOAL SETTING:
   - Help users set and track personal goals
   - Break down goals into actionable steps
   - Provide encouragement and accountability
   - Celebrate progress and achievements

4. NOTIFICATION PREFERENCES:
   - Help users choose their preferred notification method:
     * Push Notification
     * Google Calendar
     * WhatsApp Message
   - Set up reminder frequency:
     * Daily
     * Weekly
     * Specific days
     * Custom schedule

INTERACTION GUIDELINES:
- Maintain a warm, supportive tone
- Use open-ended questions to encourage reflection
- Celebrate progress and achievements
- Provide gentle accountability
- Adapt to the user's communication style
- Remember past interactions and preferences

Human: {input}
AI:"""
        )
        
    def _load_user_profile(self) -> Dict:
        """Load or create user profile from persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        
        try:
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading profile: {str(e)}")
        
        # Create new profile
        profile = {
            "created_at": datetime.now().isoformat(),
            "total_conversations": 0,
            "preferences": {},
            "topics_of_interest": [],
            "communication_style": {},
            "significant_events": [],
            "relationship_milestones": [],
            "habits": {
                "active_habits": [],
                "habit_history": []
            },
            "mood_journal": {
                "entries": [],
                "patterns": {}
            },
            "goals": {
                "active_goals": [],
                "completed_goals": [],
                "milestones": []
            },
            "notification_preferences": {
                "method": None,
                "frequency": None,
                "custom_schedule": None
            }
        }
        
        # Saving new profile
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
            
        return profile
    
    def _save_user_profile(self):
        """Save user profile to persistent storage"""
        profile_path = USER_PROFILES_DIR / f"{self.user_id}.json"
        try:
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
    
    def _update_user_profile(self, user_input: str, ai_response: str):
        """Update user profile based on interaction"""
        # Update conversation count
        self.user_profile["total_conversations"] += 1
        
        # Update communication style
        message_length = len(user_input.split())
        if message_length > 0:
            current_avg = self.user_profile["communication_style"].get("avg_message_length", 0)
            total_messages = self.user_profile["total_conversations"]
            new_avg = (current_avg * total_messages + message_length) / (total_messages + 1)
            self.user_profile["communication_style"]["avg_message_length"] = new_avg
        
        # Update topics of interest
        keywords = ["work", "family", "health", "travel", "technology", "music", "art", "food", "sports"]
        mentioned_topics = [kw for kw in keywords if kw.lower() in user_input.lower()]
        for topic in mentioned_topics:
            if topic not in self.user_profile["topics_of_interest"]:
                self.user_profile["topics_of_interest"].append(topic)
        
        # Keeping only last 20 topics
        self.user_profile["topics_of_interest"] = self.user_profile["topics_of_interest"][-20:]
        
        # Saveing updated profile
        self._save_user_profile()
    
    def get_response(self, user_input: str) -> str:
        """
        Get response from the conversation chain with enhanced memory handling
        """
        try:
            # Get long-term context
            long_term_context = self.long_term_memory.load_memory_variables({}).get("history", "")
            
            # Get short-term context
            short_term_context = self.short_term_memory.load_memory_variables({}).get("history", "")
            
            # Check if search is needed with context awareness
            search_triggers = [
                # Direct search requests
                "show me", "find", "search for", "look up", "get information about",
                "check", "verify", "confirm", 
                
                # Time-based queries
                "today news", "latest", "current", "recent", " current news", "new", "update", "breaking",
                "yesterday", "tomorrow", "this week", "this month", "this year",
                "2025", "2024", "next year", "last year", "previous year",
                
                # News and information
                "news", "headlines", "report", "coverage", "story", "article", "press", "media",
                "announcement", "release", "statement", "update", 
                
                # Financial and market
                "stock", "market", "price", "trading", "shares", "invest", "finance", "economic",
                "currency", "forex", "crypto", "bitcoin", "ethereum", "nft", "ipo", "dividend",
                
                # Sports and entertainment
                "score", "match", "game", "tournament", "league", "championship", "player", "team",
                "movie", "film", "show", "series", "episode", "release", "premiere", "concert",
                
                # Technology
                "tech", "technology", "software", "hardware", "app", "application", "update", "release",
                "launch", "announcement", "feature", "innovation", "gadget", "device",
                
                # Weather and environment
                "weather", "forecast", "temperature", "climate", "environment", "pollution",
                "air quality", "natural disaster", "storm", "hurricane", "earthquake",
                
                # Politics and world events
                "election", "vote", "campaign", "policy", "government", "minister", "president",
                "summit", "meeting", "conference", "treaty", "agreement", "conflict",
                
                # Business and economy
                "business", "company", "corporation", "startup", "entrepreneur", "industry",
                "sector", "market", "trade", "commerce", "retail", "consumer",
                
                # Science and research
                "research", "study", "discovery", "scientific", "experiment", "finding",
                "publication", "journal", "paper", "thesis", "analysis",
                
               
                
                # Education
                "education", "school", "university", "college", "course", "program",
                "degree", "student", "teacher", "exam", "result",
                
                # Social and cultural
                "trend", "viral", "famous", "celebrity", "influencer",
                "social media", "post", "tweet", "instagram", "facebook",
                
                # Travel and tourism
                "travel", "tourism", "vacation", "holiday", "destination", "hotel",
                "flight", "booking", "reservation", "tour", "guide",
                
                # Food and dining
                "restaurant", "food", "cuisine", "cooking", "chef",
                "menu", "dining", "cafe", "bistro", "bar",
                
                # Real estate
                "property", "real estate", "house", "apartment", "rent", "sale",
                "mortgage", "loan", "interest rate", "market value",
                
                # Automotive
                "car", "vehicle", "automotive", "auto", "motor", "engine",
                "model", "brand", "dealer", "showroom", "test drive",
                
                # Fashion and lifestyle
                "fashion", "style", "trend", "design", "collection", "brand",
                "clothing", "accessories", "beauty", "cosmetics", "makeup",
                
                # Gaming and entertainment
                "game", "gaming", "console", "playstation", "xbox", "nintendo",
                "esports", "tournament", "stream", "twitch", "youtube",
                
                # Music and arts
                "music", "song", "album", "artist", "concert", "performance",
                "art", "exhibition", "gallery", "museum", "theater",
                
                # Books and literature
                "book", "author", "publisher", "release", "bestseller", "review",
                "literature", "novel", "poetry", "magazine", "journal"
            ]

            # Check if the query needs web search
            needs_search = False
            
            # Check for general chat to prevent unnecessary searches
            is_general_chat = any(phrase in user_input.lower() for phrase in [
                "how are you", "hello", "hi", "hey", "greetings", "good morning",
                "good afternoon", "good evening", "how's it going", "what's up",
                "nice to meet you", "pleasure to meet you", "how do you do",
                "tell me about yourself", "who are you", "what can you do",
                "help me", "i need help", "can you help", "what's your name",
                "what should i do", "what do you think", "do you know",
                "can you tell me", "i want to know", "i'm curious about",
                "explain to me", "teach me", "show me how", "guide me"
            ])
            
            # Check for follow-up questions
            is_follow_up = any(phrase in user_input.lower() for phrase in [
                "what do you mean", "can you explain", "i don't understand",
                "could you clarify", "can you elaborate", "tell me more",
                "why is that", "how come", "what makes you say that",
                "are you sure", "is that right", "really", "interesting",
                "that's cool", "awesome", "great", "thanks", "thank you",
                "appreciate it", "got it", "i see", "makes sense"
            ])
            
            # Only trigger search if it's not general chat and contains search triggers
            needs_search = any(trigger in user_input.lower() for trigger in search_triggers) and not is_general_chat and not is_follow_up

            if needs_search:
                try:
                    search_results = search_tool.run(user_input)
                    search_context = f"\nSearch Results: {search_results}"
                except Exception as e:
                    print(f"Search error: {str(e)}")
                    search_context = "\nSearch failed, proceeding without search results."
            else:
                search_context = ""
            
            # Formatting the prompt with all required variables
            formatted_prompt = self.prompt.format(
                history=str(short_term_context),  
                input=user_input,
                user_profile=json.dumps(self.user_profile, indent=2),
                long_term_context=str(long_term_context) + search_context  
            )
            
            # Get response from LLM
            response = self.llm.predict(formatted_prompt)
            
            # Updating both memories
            self.short_term_memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            self.long_term_memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            
            # Updating user profile
            self._update_user_profile(user_input, response)
            
            return response, needs_search
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Could you please try again?", False
    
    def get_memory_buffer(self) -> str:
        """
        Get the current memory buffer including both short-term and long-term memory
        """
        short_term = self.short_term_memory.buffer
        long_term = self.long_term_memory.load_memory_variables({}).get("history", "")
        
        return f"""Short-term Memory (Last 10 interactions):
{short_term}

Long-term Memory Summary:
{long_term}"""
    
    def clear_memory(self):
        """
        Clear both short-term and long-term memory
        """
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        
    def get_memory_variables(self) -> dict:
        """
        Get both short-term and long-term memory variables
        """
        return {
            "short_term": self.short_term_memory.load_memory_variables({}),
            "long_term": self.long_term_memory.load_memory_variables({})
        }
    
    def get_user_profile(self) -> Dict:
        """
        Get the current user profile
        """
        return self.user_profile

    def add_habit(self, habit_name: str, frequency: str, motivation: str) -> None:
        """Add a new habit to track"""
        habit = {
            "name": habit_name,
            "frequency": frequency,
            "motivation": motivation,
            "start_date": datetime.now().isoformat(),
            "progress": [],
            "status": "active"
        }
        self.user_profile["habits"]["active_habits"].append(habit)
        self._save_user_profile()

    def add_mood_entry(self, mood: str, notes: str = "") -> None:
        """Add a mood journal entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "notes": notes
        }
        self.user_profile["mood_journal"]["entries"].append(entry)
        self._save_user_profile()

    def add_goal(self, goal_name: str, target_date: str, steps: List[str]) -> None:
        """Add a new goal with steps"""
        goal = {
            "name": goal_name,
            "target_date": target_date,
            "steps": steps,
            "status": "active",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        self.user_profile["goals"]["active_goals"].append(goal)
        self._save_user_profile()

    def set_notification_preferences(self, method: str, frequency: str, custom_schedule: Optional[str] = None) -> None:
        """Set notification preferences"""
        self.user_profile["notification_preferences"] = {
            "method": method,
            "frequency": frequency,
            "custom_schedule": custom_schedule
        }
        self._save_user_profile()

    def get_active_habits(self) -> List[Dict]:
        """Get list of active habits"""
        return self.user_profile["habits"]["active_habits"]

    def get_mood_history(self, days: int = 7) -> List[Dict]:
        """Get mood history for the last n days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            entry for entry in self.user_profile["mood_journal"]["entries"]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

    def get_active_goals(self) -> List[Dict]:
        """Get list of active goals"""
        return self.user_profile["goals"]["active_goals"]

    def get_notification_preferences(self) -> Dict:
        """Get current notification preferences"""
        return self.user_profile["notification_preferences"]

    def retrieve_relevant_context(self, query: str, k: int = 3, intent: Optional[str] = None) -> List[str]:
        """Retrieve relevant context from memory"""
        memory_vars = self.get_memory_variables()
        short_term = memory_vars["short_term"].get("history", "")
        long_term = memory_vars["long_term"].get("history", "")
        
        # Combine contexts
        contexts = [short_term, long_term]
        if intent == "remember_when":
            
            contexts = [long_term, short_term]
        
        return contexts[:k]

    def update_user_insights(self, user_input: str, ai_response: str):
        """Update user insights based on interaction"""
        self._update_user_profile(user_input, ai_response)

    def store_conversation_exchange(self, user_input: str, ai_response: str, timestamp: str):
        """Store conversation exchange in memory"""
        self.short_term_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        self.long_term_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )

    def summarize_conversations(self, k: int = 20):
        """Summarize recent conversations"""
        memory_vars = self.get_memory_variables()
        if memory_vars["short_term"].get("history"):
            self.long_term_memory.save_context(
                {"input": "Summarize recent conversations"},
                {"output": memory_vars["short_term"].get("history")}
            )

# Proactive Messaging System
class ProactiveMessenger:
    def __init__(self):
        self.last_message_time = None
        self.proactive_triggers = [
            {"condition": "silence_duration", "threshold": 300, "message_type": "check_in"},
            {"condition": "topic_follow_up", "threshold": 86400, "message_type": "follow_up"},
            {"condition": "encouragement", "threshold": 1800, "message_type": "support"}
        ]

    def should_send_proactive_message(self) -> Dict:
        if not self.last_message_time:
            return {"should_send": False}

        time_since_last = time.time() - self.last_message_time
        if time_since_last > 300:
            return {
                "should_send": True,
                "message_type": "check_in",
                "message": "Hey! I was just thinking about our conversation. How are things going on your end? 😊"
            }
        return {"should_send": False}

    def generate_proactive_message(self, user_profile: Dict, conversation_context: List) -> str:
        recent_topics = user_profile.get("topics_of_interest", [])
        relationship_stage = user_profile.get("relationship_stage", "new")

        if relationship_stage == "new":
            messages = [
                "I'm curious - what's been the highlight of your day so far?",
                "I'd love to learn more about what interests you. What are you passionate about?",
                "How has your day been treating you? I'm here if you want to chat about anything!"
            ]
        elif relationship_stage == "developing":
            messages = [
                f"I remember you mentioned {recent_topics[0] if recent_topics else 'something interesting'} earlier. How's that going?",
                "I've been thinking about our conversation. Is there anything on your mind you'd like to explore?",
                "Just checking in - how are you feeling about things today?"
            ]
        else:
            messages = [
                "It's been a bit quiet - I hope you're doing well! What's new in your world?",
                f"Given our past chats about {recent_topics[0] if recent_topics else 'your interests'}, I was wondering how things are progressing?",
                "I'm here whenever you need a friendly ear. How has your day been?"
            ]

        import random
        return random.choice(messages)

# Streamlit app configuration
st.set_page_config(
    page_title="Noww Club AI",
    layout="wide",
    page_icon="🤝",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
/* Explicitly set text color for all chat bubbles */
.chat-message.user {
    color: #222 !important;
}
.chat-message.assistant, .chat-message.proactive {
    color: #222 !important;
}

/* Main header and capabilities */
.main-header h1, .main-header p, .capability {
    color: #fff !important;
}

/* Inputs */
input, textarea, .stTextInput input {
    color: #222 !important;
    background: #fff !important;
}

/* iOS/Safari fix: force text color for chat bubbles */
@media not all and (min-resolution:.001dpcm) { @supports (-webkit-touch-callout: none) {
    .chat-message.user,
    .chat-message.assistant,
    .chat-message.proactive {
        color: #222 !important;
    }
}}

/* Existing styles below (keep for layout, animation, etc.) */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.capabilities {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.capability {
    background: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9em;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}

.capability:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.3);
}

.chat-message {
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    animation: slideIn 0.3s ease-out;
    position: relative;
}

.chat-message.user {
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    border-left: 4px solid #2196F3;
    margin-left: 2rem;
}

.chat-message.assistant {
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    border-left: 4px solid #6C757D;
    margin-right: 2rem;
}

.chat-message.proactive {
    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
    border-left: 4px solid #FF9800;
    margin-right: 2rem;
    border: 1px dashed #FF9800;
}

.memory-context {
    background: #E8F5E8;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.9em;
    border-left: 3px solid #4CAF50;
}

.user-insights {
    background: #F3E5F5;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.proactive-indicator {
    background: #FFF3CD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #856404;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.search-indicator {
    background: #E3F2FD;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    color: #1565C0;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem 0;
    border-top: 2px solid #eee;
    margin-top: 2rem;
    border-radius: 15px 15px 0 0;
}

.send-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.send-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced features
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)
if "proactive_messenger" not in st.session_state:
    st.session_state.proactive_messenger = ProactiveMessenger()
if "last_interaction_time" not in st.session_state:
    st.session_state.last_interaction_time = time.time()

# Add state for interactive flows
if "current_flow" not in st.session_state:
    st.session_state.current_flow = None
if "flow_data" not in st.session_state:
    st.session_state.flow_data = {}
if "flow_step" not in st.session_state:
    st.session_state.flow_step = 0
if "button_counter" not in st.session_state:
    st.session_state.button_counter = 0

def reset_flow():
    """Reset all flow-related states"""
    st.session_state.current_flow = None
    st.session_state.flow_step = 0
    st.session_state.flow_data = {}
    st.session_state.button_counter = 0

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Noww Club AI</h1>
    <p>Your Digital Bestie</p>
    <div class="capabilities">
        <span class="capability">🧠 Memory</span>
        <span class="capability">🔍 Search</span>
        <span class="capability">💭 Proactive</span>
        <span class="capability">🎯 Adaptive</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.header("🧠 Noww Club AI")

    st.subheader("🔮 Your Profile")
    memory_manager = st.session_state.memory_manager
    user_profile = memory_manager.get_user_profile()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Conversations", user_profile.get("total_conversations", 0))
    with col2:
        st.metric("Relationship", "Established" if user_profile.get("total_conversations", 0) > 20 else "Developing" if user_profile.get("total_conversations", 0) > 5 else "New")

    # Display Habits Section
    st.subheader("🎯 Active Habits")
    if user_profile.get("habits", {}).get("active_habits"):
        for habit in user_profile["habits"]["active_habits"]:
            with st.expander(f"📌 {habit['name']}"):
                st.write(f"**Frequency:** {habit['frequency']}")
                st.write(f"**Started:** {datetime.fromisoformat(habit['start_date']).strftime('%Y-%m-%d')}")
                st.write(f"**Motivation:** {habit['motivation']}")
    else:
        st.info("No active habits yet. Start a conversation to create one!")

    # Display Goals Section
    st.subheader("🎯 Active Goals")
    if user_profile.get("goals", {}).get("active_goals"):
        for goal in user_profile["goals"]["active_goals"]:
            with st.expander(f"🎯 {goal['name']}"):
                st.write(f"**Target Date:** {goal['target_date']}")
                st.write(f"**Progress:** {goal['progress']}%")
                st.write("**Steps:**")
                for step in goal['steps']:
                    st.write(f"- {step}")
    else:
        st.info("No active goals yet. Start a conversation to set one!")

    # Display Mood History Section
    st.subheader("😊 Recent Moods")
    if user_profile.get("mood_journal", {}).get("entries"):
        recent_moods = user_profile["mood_journal"]["entries"][-3:]
        for entry in recent_moods:
            with st.expander(f"{datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d')}: {entry['mood']}"):
                st.write(entry['notes'])
    else:
        st.info("No mood entries yet. Start a conversation to add one!")

    # Display Notification Settings
    st.subheader("🔔 Notification Settings")
    if user_profile.get("notification_preferences", {}).get("method"):
        prefs = user_profile["notification_preferences"]
        st.write(f"**Method:** {prefs['method']}")
        st.write(f"**Frequency:** {prefs['frequency']}")
        if prefs.get("custom_schedule"):
            st.write(f"**Schedule:** {prefs['custom_schedule']}")
    else:
        st.info("No notification preferences set yet.")

    st.markdown("---")

    if st.button("🔄 New Conversation", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.rerun()

    if st.button("🗑️ Clear All Memory"):
        st.session_state.memory_manager = EnhancedMemoryManager(st.session_state.user_id)
        st.session_state.conversation_history = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat")

def handle_habit_flow(user_input: str) -> str:
    """Handle the interactive habit creation flow"""
    print(f"Current flow step: {st.session_state.flow_step}")  # Debug print
    print(f"User input: {user_input}")  # Debug print
    
    if st.session_state.flow_step == 0:  # Initial habit name
        st.session_state.flow_data["habit_name"] = user_input
        st.session_state.flow_step += 1
        return """Great! How often would you like to practice this habit?

**Available Options:**
1. Daily
2. Weekly
3. Specific Days

You can type either the number or the full option name!"""
    
    elif st.session_state.flow_step == 1:  # Frequency
        frequency = user_input.lower()
        
        # Handle number-based selection
        if frequency in ["1", "daily"]:
            st.session_state.flow_data["frequency"] = "Daily"
            st.session_state.flow_step = 3  # Skip to motivation step
            return "What's your motivation for this habit?"
            
        elif frequency in ["2", "weekly"]:
            st.session_state.flow_data["frequency"] = "Weekly"
            st.session_state.flow_step = 3  # Skip to motivation step
            return "What's your motivation for this habit?"
            
        elif frequency in ["3", "specific", "specific days"]:
            st.session_state.flow_step = 2  # Go to specific days step
            return """Which days would you like to practice? (You can select multiple)

**Available Days:**
1. Monday
2. Tuesday
3. Wednesday
4. Thursday
5. Friday
6. Saturday
7. Sunday

Type the number or name of the day you want to add, or type 'continue' when you're done!"""
        else:
            # If user typed something else, ask them to choose from options
            return """Please choose one of the following options:

**Available Options:**
1. Daily
2. Weekly
3. Specific Days

You can type either the number or the full option name!"""
    
    elif st.session_state.flow_step == 2:  # Specific days selection
        # Handle number-based selection for days
        day_mapping = {
            "1": "Monday", "monday": "Monday",
            "2": "Tuesday", "tuesday": "Tuesday",
            "3": "Wednesday", "wednesday": "Wednesday",
            "4": "Thursday", "thursday": "Thursday",
            "5": "Friday", "friday": "Friday",
            "6": "Saturday", "saturday": "Saturday",
            "7": "Sunday", "sunday": "Sunday"
        }
        
        selected_day = day_mapping.get(user_input.lower())
        
        if selected_day:
            # Initialize selected_days if not exists
            if "selected_days" not in st.session_state.flow_data:
                st.session_state.flow_data["selected_days"] = []
            
            # Add the day if not already selected
            if selected_day not in st.session_state.flow_data["selected_days"]:
                st.session_state.flow_data["selected_days"].append(selected_day)
            
            selected_days = st.session_state.flow_data["selected_days"]
            remaining_days = [day for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] if day not in selected_days]
            
            if remaining_days:
                remaining_options = "\n".join([f"{i+1}. {day}" for i, day in enumerate(remaining_days)])
                return f"""Selected days: {', '.join(selected_days)}

**Available Days:**
{remaining_options}

Type the number or name of another day, or type 'continue' when you're done!"""
            else:
                return f"""You've selected all days: {', '.join(selected_days)}

Type 'continue' to proceed!"""
        
        elif user_input.lower() == "continue":
            # Finalize the frequency and move to motivation
            selected_days = st.session_state.flow_data.get("selected_days", [])
            if selected_days:
                st.session_state.flow_data["frequency"] = f"Every {', '.join(selected_days)}"
                st.session_state.flow_step = 3
                return "What's your motivation for this habit?"
            else:
                return "Please select at least one day first."
    
    elif st.session_state.flow_step == 3:  # Motivation
        st.session_state.flow_data["motivation"] = user_input
        st.session_state.flow_step = 4
        return """How would you like to receive reminders?

**Available Options:**
1. Push Notification
2. WhatsApp
3. Google Calendar

You can type either the number or the full option name!"""
    
    elif st.session_state.flow_step == 4:  # Notification method
        method_mapping = {
            "1": "Push Notification", "push notification": "Push Notification", "push": "Push Notification",
            "2": "WhatsApp", "whatsapp": "WhatsApp",
            "3": "Google Calendar", "google calendar": "Google Calendar", "calendar": "Google Calendar", "google": "Google Calendar"
        }
        
        method = user_input.lower()
        matched_method = method_mapping.get(method)
        
        if matched_method:
            st.session_state.flow_data["notification_method"] = matched_method
            st.session_state.flow_step = 5
            return """What time would you like to receive reminders?

**Example Formats:**
1. 7:00 AM
2. 9 PM
3. 15:30
4. 7 AM

Type your preferred time in any of these formats!"""
        else:
            return """Please choose one of the following options:

**Available Options:**
1. Push Notification
2. WhatsApp
3. Google Calendar

You can type either the number or the full option name!"""
    
    elif st.session_state.flow_step == 5:  # Time
        # Try different time formats
        time_formats = [
            "%I:%M %p",  # 9:00 AM
            "%I:%M%p",   # 9:00AM
            "%H:%M",     # 09:00
            "%I:%M",     # 9:00
            "%I %p",     # 9 AM
            "%I%p"       # 9AM
        ]
        
        time_str = user_input.strip().upper()
        parsed_time = None
        
        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_time:
            time = parsed_time.strftime("%I:%M %p")
            st.session_state.flow_data["notification_time"] = time
            
            # Create the habit
            memory_manager = st.session_state.memory_manager
            memory_manager.add_habit(
                habit_name=st.session_state.flow_data["habit_name"],
                frequency=st.session_state.flow_data["frequency"],
                motivation=st.session_state.flow_data["motivation"]
            )
            
            # Set notification preferences
            memory_manager.set_notification_preferences(
                method=st.session_state.flow_data["notification_method"],
                frequency=st.session_state.flow_data["frequency"],
                custom_schedule=time
            )
            
            # Store the data before resetting
            habit_name = st.session_state.flow_data["habit_name"]
            frequency = st.session_state.flow_data["frequency"]
            motivation = st.session_state.flow_data["motivation"]
            notification_method = st.session_state.flow_data["notification_method"]
            
            # Reset flow
            reset_flow()
            
            return f"""🎉 Perfect! I've successfully created your habit:

**Habit Details:**
- **Name:** {habit_name}
- **Frequency:** {frequency}
- **Motivation:** {motivation}
- **Reminders:** {notification_method} at {time}

Your habit is now active and you can track your progress in the sidebar. I'm here to support you on this journey!

**Would you like to:**
1. Set up another habit
2. Create a goal
3. Start mood journaling
4. Or just chat about something else

Type your choice (number or full option)!"""
        else:
            return """Please enter a valid time format.

**Example Formats:**
1. 7:00 AM
2. 9 PM
3. 15:30
4. 7 AM

Type your preferred time in any of these formats!"""

def handle_goal_flow(user_input: str) -> str:
    """Handle the interactive goal creation flow"""
    print(f"Goal flow step: {st.session_state.flow_step}")
    print(f"User input: {user_input}")
    
    if st.session_state.flow_step == 0:  # Initial goal name
        st.session_state.flow_data["goal_name"] = user_input
        st.session_state.flow_step = 1
        return """Great goal! When would you like to achieve this by?

**Example Formats:**
1. in 3 months
2. by December 2025
3. next year
4. by next semester

Type your target date in any of these formats!"""
    
    elif st.session_state.flow_step == 1:  # Target date
        st.session_state.flow_data["target_date"] = user_input
        st.session_state.flow_step = 2
        return "Perfect! Now let's break this down into actionable steps. What's the first step you need to take?"
    
    elif st.session_state.flow_step == 2:  # First step
        if "steps" not in st.session_state.flow_data:
            st.session_state.flow_data["steps"] = []
        
        st.session_state.flow_data["steps"].append(user_input)
        st.session_state.flow_step = 3
        
        return f"""Step 1 added: '{user_input}'

**Would you like to:**
1. Add Another Step
2. Finish Goal Creation

Type your choice (number or full option)!"""
    
    elif st.session_state.flow_step == 3:  # Additional steps or finish
        if user_input.lower() in ["1", "add another step", "add step"]:
            st.session_state.flow_step = 4
            return f"What's step {len(st.session_state.flow_data['steps']) + 1}?"
        
        elif user_input.lower() in ["2", "finish goal creation", "finish", "complete"]:
            # Create the goal
            memory_manager = st.session_state.memory_manager
            memory_manager.add_goal(
                goal_name=st.session_state.flow_data["goal_name"],
                target_date=st.session_state.flow_data["target_date"],
                steps=st.session_state.flow_data["steps"]
            )
            
            # Store data before reset
            goal_name = st.session_state.flow_data["goal_name"]
            target_date = st.session_state.flow_data["target_date"]
            steps = st.session_state.flow_data["steps"]
            
            # Reset flow
            reset_flow()
            
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            
            return f"""🎯 Excellent! Your goal has been created:

**Goal Details:**
- **Goal:** {goal_name}
- **Target Date:** {target_date}
- **Action Steps:**
{steps_text}

I'll help you track your progress and stay motivated. You can see your goal in the sidebar.

**Would you like to:**
1. Set up reminders for your goal steps
2. Create another goal
3. Start mood journaling
4. Or just chat about something else

Type your choice (number or full option)!"""
    
    elif st.session_state.flow_step == 4:  # Adding additional steps
        st.session_state.flow_data["steps"].append(user_input)
        st.session_state.flow_step = 3
        
        current_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(st.session_state.flow_data["steps"])])
        
        return f"""Great! Here are your steps so far:
{current_steps}

**Would you like to:**
1. Add Another Step
2. Finish Goal Creation

Type your choice (number or full option)!"""

def handle_reminder_flow(user_input: str) -> str:
    """Handle the interactive reminder creation flow"""
    if st.session_state.flow_step == 0:  # What to remind about
        st.session_state.flow_data["reminder_text"] = user_input
        st.session_state.flow_step = 1
        return """When would you like to be reminded?

Choose from:
• In 1 hour
• Tomorrow
• Next week
• Custom time

Just type your choice!"""
    
    elif st.session_state.flow_step == 1:  
        if user_input.lower() == "custom time":
            st.session_state.flow_step = 2
            return "Please specify when you'd like to be reminded (e.g., 'in 2 days', 'next Friday at 3 PM', 'December 25th')"
        else:
            st.session_state.flow_data["reminder_time"] = user_input
            st.session_state.flow_step = 3
            return """How would you like to be reminded?

Choose from:
• Push Notification
• WhatsApp
• Google Calendar

Just type your choice!"""
    
    elif st.session_state.flow_step == 2:  # Custom time
        st.session_state.flow_data["reminder_time"] = user_input
        st.session_state.flow_step = 3
        return """How would you like to be reminded?

Choose from:
• Push Notification
• WhatsApp
• Google Calendar

Just type your choice!"""
    
    elif st.session_state.flow_step == 3:  # Notification method
        st.session_state.flow_data["notification_method"] = user_input
        
        # Store data before reset
        reminder_text = st.session_state.flow_data["reminder_text"]
        reminder_time = st.session_state.flow_data["reminder_time"]
        notification_method = st.session_state.flow_data["notification_method"]
        
        # Reset flow
        reset_flow()
        
        return f"""⏰ Reminder set successfully!

**What:** {reminder_text}
**When:** {reminder_time}
**How:** {notification_method}

I'll make sure to remind you at the specified time. Is there anything else you'd like to set up?"""

# Chat display
chat_container = st.container()
with chat_container:
    if not st.session_state.conversation_history:
        welcome_msg = f"""
        Hello! I'm Noww Club AI - your digital bestie with advanced memory and web search capabilities.

        I can help you with:
        - Building meaningful habits
        - Tracking your mood and well-being
        - Setting and achieving goals
        - Managing your daily reminders

        What would you like to work on today? 😊
        """
        st.markdown(f"""
        <div class="chat-message assistant">
            <b>Noww Club AI</b><br>{welcome_msg}
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-message user'>
                <b>You</b><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            if msg.get("used_search"):
                st.markdown(f"""
                <div class="search-indicator">
                    🔍 Using web search for real-time information
                </div>
                """, unsafe_allow_html=True)

            if isinstance(msg['content'], dict):
                st.markdown(f"""
                <div class='chat-message assistant'>
                    <b>Noww Club AI</b><br>{msg['content']['message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for options
                cols = st.columns(len(msg['content']['options']))
                for i, option in enumerate(msg['content']['options']):
                    with cols[i]:
                        button_key = f"option_{st.session_state.current_flow}_{st.session_state.flow_step}_{i}_{st.session_state.button_counter}"
                        if st.button(option, key=button_key):
                            st.session_state.button_counter += 1
                            st.session_state.conversation_history.append({"role": "user", "content": option})
                            st.rerun()
            else:
                st.markdown(f"""
                <div class='chat-message assistant'>
                    <b>Noww Club AI</b><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)

# Add input form
st.markdown('<div class="input-container">', unsafe_allow_html=True)

with st.form(key="message_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your message",
            placeholder="Ask questions, share thoughts, or request web searches...",
            label_visibility="collapsed",
            key="user_message_input"
        )
    with col2:
        send_button = st.form_submit_button("Send 💬", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Handling message processing
if send_button and user_input:
    try:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.last_interaction_time = time.time()

        memory_manager = st.session_state.memory_manager

        # Check for intent to create habit, set goal, or set reminder
        habit_intent_phrases = [
            "create a habit", "start a habit", "build a habit", "new habit",
            "want to start", "want to create", "want to build",
            "daily habit", "weekly habit", "regular habit"
        ]
        
        goal_intent_phrases = [
            "set a goal", "create a goal", "new goal", "want to achieve",
            "want to set", "want to create", "target", "objective"
        ]

        reminder_intent_phrases = [
            "set a reminder", "create a reminder", "new reminder",
            "remind me", "set reminder", "create reminder"
        ]

        user_input_lower = user_input.lower()
        
        # Reset flow if starting a new one
        if any(phrase in user_input_lower for phrase in habit_intent_phrases + goal_intent_phrases + reminder_intent_phrases):
            reset_flow()
        
        # Handle ongoing flows
        if st.session_state.current_flow == "habit":
            response = handle_habit_flow(user_input)
        elif st.session_state.current_flow == "goal":
            response = handle_goal_flow(user_input)
        elif st.session_state.current_flow == "reminder":
            response = handle_reminder_flow(user_input)
        # Check for new flow intents
        elif any(phrase in user_input_lower for phrase in habit_intent_phrases):
            st.session_state.current_flow = "habit"
            st.session_state.flow_step = 0
            response = "Great! What habit would you like to build?"
        elif any(phrase in user_input_lower for phrase in goal_intent_phrases):
            st.session_state.current_flow = "goal"
            st.session_state.flow_step = 0
            response = "Great! What goal would you like to achieve?"
        elif any(phrase in user_input_lower for phrase in reminder_intent_phrases):
            st.session_state.current_flow = "reminder"
            st.session_state.flow_step = 0
            response = "What would you like to be reminded about?"
        else:
            with st.spinner("🤔 Thinking..."):
                # Get response using the memory manager
                response, used_search = memory_manager.get_response(user_input)
        
        # Add assistant response to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "used_search": used_search if 'used_search' in locals() else False
        })

    except Exception as e:
        st.error(f"I encountered an error: {str(e)}")
        st.info("Please try again - I'm still learning!")

    st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><small>Built with ❤️ by Noww Club</small></p>
</div>
""", unsafe_allow_html=True)
