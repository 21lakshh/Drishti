from dotenv import load_dotenv
import logging 
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, TurnHandlingOptions
from livekit.plugins import (
    noise_cancellation,
    sarvam,
    )
from livekit.agents.llm import function_tool
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass, field
from typing import List, Optional, Annotated
from pydantic import Field
import os
import json
import base64
import httpx
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from qdrant_client import AsyncQdrantClient, models
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")
_http_client = httpx.AsyncClient(base_url=MODEL_SERVER_URL, timeout=60.0)

load_dotenv()

@dataclass
class UserData:
    object_to_find: Optional[str] = None
    user_location: Optional[str] = None
    object_found: bool = False
    object_image: Optional[str] = None
    detected_box: Optional[List[float]] = None
    prev_agent: Optional[Agent] = None
    preferred_language: Optional[str] = None
    preferred_language_code: Optional[str] = "en-IN"
    agents: dict[str, Agent] = field(default_factory=dict)

    def summarize(self) -> str:
        data = {
            "object_to_find": self.object_to_find or "unknown",
            "user_location": self.user_location or "unknown",
            "object_found": self.object_found,
            "object_image": self.object_image or "no image",
            "prev_agent": self.prev_agent.__class__.__name__ if self.prev_agent else "no previous agent"
        }
        return json.dumps(data, indent=2)

RunContext_T = RunContext[UserData]


class BaseAgent(Agent):
    async def on_enter(self, generate_reply: bool = True) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        language_instruction = ""
        if userdata.preferred_language and userdata.preferred_language.lower() != "english":
            language_instruction = f" IMPORTANT: Always respond in {userdata.preferred_language} language, not English."
        
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}.{language_instruction}",
        )
        await self.update_chat_ctx(chat_ctx)
        if generate_reply:
            self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: Optional[RunContext_T] = None) -> tuple[Agent, str]:
        if context is None:
            # Called from lifecycle method, use self.session
            userdata = self.session.userdata
            current_agent = self.session.current_agent
        else:
            # Called from function tool, use provided context
            userdata = context.userdata
            current_agent = context.session.current_agent
        
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."

class Greeting(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You're a calm, real-time smart voice navigator based inside a user's house. "
                "Collect the user's target item, user's location and preferred language, confirm what you heard. "
                "Once you have ALL THREE pieces of information (object, location, AND language), call the start_detection tool to begin scanning. "
                "CRITICAL: Once user provides their preferred language, ALWAYS respond in that language (not English). "
                "Keep responses short, actionable, and conversational."
            ),
        )
    
    @function_tool()
    async def update_object_to_find(
        self, 
        object_to_find: Annotated[str, Field(description="The object the user wants to find")], 
        context: RunContext_T) -> str:
        """Called when user provides the object they want to find"""
        userdata = context.userdata
        target = object_to_find.strip()
        userdata.object_to_find = target

        missing = []
        if not userdata.user_location:
            missing.append("location")
        if not userdata.preferred_language_code or userdata.preferred_language_code == "en-IN":
            missing.append("preferred language")
        
        if missing:
            return f"Locked onto {target}. Please also provide your {' and '.join(missing)}."
        
        return f"Locked onto {target}."

    @function_tool()
    async def update_user_location(
        self, 
        user_location: Annotated[str, Field(description="The location of the user")], 
        context: RunContext_T) -> str:
        """Called when user provides their location"""
        userdata = context.userdata
        location = user_location.strip()
        userdata.user_location = location

        missing = []
        if not userdata.object_to_find:
            missing.append("object to find")
        if not userdata.preferred_language_code or userdata.preferred_language_code == "en-IN":
            missing.append("preferred language")
        
        if missing:
            return f"Noted—{location}. Please also tell me the {' and '.join(missing)}."
        
        return f"Noted—{location}."
    
    @function_tool()
    async def update_user_preferred_language(
        self, 
        preferred_language: Annotated[str, Field(description="The user's preferred language for communication")], 
        context: RunContext_T) -> str:
        """Called when user provides their preferred language"""
        userdata = context.userdata
        language = preferred_language.strip()

        # Map common language names to Sarvam API language codes
        language_mapping = {
            "english": "en-IN",
            "hindi": "hi-IN",
            "bengali": "bn-IN",
            "kannada": "kn-IN",
            "malayalam": "ml-IN",
            "marathi": "mr-IN",
            "odia": "od-IN",
            "punjabi": "pa-IN",
            "tamil": "ta-IN",
            "telugu": "te-IN",
            "gujarati": "gu-IN",
        }
        language_code = language_mapping.get(language.lower(), "unknown")  # default to "unknown" if not found for API to detect
        userdata.preferred_language = language
        userdata.preferred_language_code = language_code

        missing = []
        if not userdata.object_to_find:
            missing.append("object to find")
        if not userdata.user_location:
            missing.append("location")
        
        if missing:
            return f"Got it, I'll communicate in {language}. Please also provide your {' and '.join(missing)}."
        
        return f"Got it, I'll communicate in {language}."
    
    @function_tool()
    async def start_detection(
        self, 
        context: RunContext_T) -> tuple[Agent, str] | str:
        """Call this to start object detection when all required information (object, location, language) has been collected. Only call this when you have confirmed all three pieces of information from the user."""
        userdata = context.userdata
        
        # Validate all required fields are present
        if not userdata.object_to_find or not userdata.user_location or not userdata.preferred_language_code:
            missing = []
            if not userdata.object_to_find:
                missing.append("object to find")
            if not userdata.user_location:
                missing.append("location")
            if not userdata.preferred_language_code or userdata.preferred_language_code == "en-IN":
                missing.append("preferred language")
            return f"I still need: {', '.join(missing)}. Please provide this information first."
        
        next_agent, _ = await self._transfer_to_agent("object_detection", context)
        return next_agent, (
            f"Perfect! Scanning for {userdata.object_to_find} in {userdata.user_location} now."
        )

class ObjectDetectionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You handle rapid object detection. Wait for the detection results. "
                "After detection, report what was found and offer TWO options: estimate distance (if found) or check the knowledge base. "
                "CRITICAL: You are DETECTION ONLY. You have absolutely NO ability to measure distances or depth. "
                "NEVER state, guess, or approximate any distance value — not even casually. "
                "Distance measurement is exclusively handled by the DepthEstimationAgent. "
                "When calling to_depth_estimation, respond ONLY with a short handoff phrase like 'On it!' or 'Measuring now.' — nothing else. "
                "CRITICAL: Always respond in the user's preferred language. "
                "Do not speak until detection is complete."
            ),
        )
        logger.info("ObjectDetectionAgent ready — inference via model server.")
    
    async def on_enter(self) -> None:
        await super().on_enter(generate_reply=False)
        
        # Run detection and add result to chat context for LLM to translate
        message = await self._run_detection()
        
        # Add detection result to chat context
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=f"Detection complete. Result: {message}. Now communicate this result to the user in their preferred language."
        )
        await self.update_chat_ctx(chat_ctx)
        
        # Let LLM generate translated response
        self.session.generate_reply(tool_choice="auto")
     
    async def _run_detection(self, context: Optional[RunContext_T] = None) -> str:
        userdata = self.session.userdata if context is None else context.userdata
        target = userdata.object_to_find
        image_path = userdata.object_image

        if not target:
            return "I don't know what object to look for."
        if not image_path:
            return "No image available for detection."

        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            response = await _http_client.post(
                "/detect",
                json={"image_b64": image_b64, "target": target},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Detection service error: {e}", exc_info=True)
            return "The detection service is unavailable. Please try again."

        logger.info(f"Detection result: {data}")

        if data["found"]:
            userdata.object_found = True
            userdata.detected_box = data["box"]
            return f"I found {data['best_match']}, which looks like your {target}. Want me to estimate the distance to it?"

        userdata.object_found = False
        names = data.get("detected_objects", [])
        if names:
            return f"I see {', '.join(names[:3])}, but nothing matching {target}. Should I check the knowledge base?"
        return "I didn't spot any objects. Should I check the knowledge base?"
    
    @function_tool()
    async def to_depth_estimation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to estimate distance to the detected object.
        Use when user says yes, sure, please do, go ahead, or similar confirmation.
        After calling this, say ONLY a short handoff phrase like 'Measuring now!' or 'On it!'.
        NEVER mention a distance value — you have no ability to measure depth."""
        logger.info("Function tool 'to_depth_estimation' called - transferring to DepthEstimationAgent")
        next_agent, _ = await self._transfer_to_agent("depth_estimation", context)
        return (
            next_agent,
            "Transfer to DepthEstimationAgent initiated. "
            "Respond with ONLY a short handoff phrase such as 'Measuring now!' or 'On it!' — "
            "do NOT state any distance value whatsoever. Distance data is unavailable at this stage.",
        )
    
    @function_tool()
    async def to_rag(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to check the knowledge base for object location. Use when object is not found or user asks to search knowledge base."""
        logger.info("Function tool 'to_rag' called - transferring to RAGAgent")
        return await self._transfer_to_agent("rag", context)

class RAGAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful navigation assistant. "
                "You have access to a building knowledge base. "
                "Use the provided context to answer the user's question about where their object might be located relative to their current location. "
                "If the context provides specific directions, give them clearly. "
                "If the context does not contain the answer, admit that you don't know and suggest checking a different room manually. "
                "CRITICAL: Always respond in the user's preferred language."
            ),
        )
        self.qdrant_client = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "drishti")
        self.user_id_filter = 1  # Hardcoded for now, later this will be adaded from livekit room metadata field

    async def on_enter(self) -> None:
        await super().on_enter(generate_reply=False)
        
        # Perform retrieval
        context_message = await self._retrieve_context()
        
        # Inject retrieved knowledge into chat context
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context_message}\n\nUse this information to guide the user."
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="auto")

    async def _retrieve_context(self) -> str:
        userdata = self.session.userdata
        query_text = f"Where is {userdata.object_to_find}? I am currently at {userdata.user_location}."
        
        logger.info(f"RAG Query: {query_text}")

        try:
            # 1. Generate Embedding
            embedding_response = await self.openai_client.embeddings.create(
                input=query_text,
                model="text-embedding-3-small"
            )
            query_vector = embedding_response.data[0].embedding

            # 2. Search Qdrant (using query_points)
            search_response = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=3,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(
                                value=self.user_id_filter,
                            ),
                        )
                    ]
                ),
            )
            
            # Extract points
            results = search_response.points

            if not results:
                return "No relevant information found in the knowledge base."

            # 3. Format Results
            context_parts = []
            for hit in results:
                text = hit.payload.get("text", "")
                context_parts.append(f"- {text}")
            
            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"RAG Retrieval failed: {e}", exc_info=True)
            return "Error retrieving information from database."

    @function_tool()
    async def search_new_object(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to search for a different object or start over."""
        logger.info("Function tool 'search_new_object' called - resetting and transferring to Greeting")
        context.userdata.object_to_find = None
        context.userdata.user_location = None
        context.userdata.object_found = False
        return await self._transfer_to_agent("greeter", context)

    @function_tool()
    async def end_session(self, context: RunContext_T) -> str:
        """Call this when the user is satisfied with the information or wants to stop."""
        return "Glad I could help. Safe travels!"

class DepthEstimationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You handle distance estimation to detected objects. "
                "Your FIRST and ONLY action must be to call the `run_depth_estimation` tool — do NOT generate any speech before calling it. "
                "After the tool returns the exact distance, communicate it clearly to the user in their preferred language. "
                "Then ask if they need further assistance or want to search for another object. "
                "CRITICAL: Always respond in the user's preferred language."
            ),
        )
        logger.info("DepthEstimationAgent ready — inference via model server.")

    async def on_enter(self) -> None:
        await super().on_enter(generate_reply=False)

        userdata: UserData = self.session.userdata
        obj = userdata.object_to_find or "the object"
        await self.session.say(
            f"Got it! Calculating the distance to your {obj}, please hold on for a moment...",
            allow_interruptions=False,
            add_to_chat_ctx=True,
        )

        self.session.generate_reply(tool_choice="required")

    @function_tool()
    async def run_depth_estimation(self, context: RunContext_T) -> str:
        """Runs depth estimation on the detected object and returns the exact distance in meters.
        Always call this immediately upon entering — never speak before calling this tool."""
        logger.info("Function tool 'run_depth_estimation' called")
        return await self._estimate_depth(context)

    async def _estimate_depth(self, context: Optional[RunContext_T] = None) -> str:
        """Run depth estimation via model server."""
        userdata = self.session.userdata if context is None else context.userdata
        object_to_find = userdata.object_to_find
        image_path = userdata.object_image
        box = userdata.detected_box  # [x1, y1, x2, y2]

        logger.info(f"Starting depth estimation for {object_to_find}")

        if not object_to_find or not image_path or not box:
            logger.warning("Missing required data for depth estimation")
            return "I need a confirmed object detection to gauge distance."

        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            response = await _http_client.post(
                "/depth",
                json={"image_b64": image_b64, "box": box, "object_name": object_to_find},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Depth service error: {e}", exc_info=True)
            return "I'm having trouble reading the depth sensors right now."

        logger.info(f"Depth result: {data}")
        return data["message"]
    
    @function_tool()
    async def search_new_object(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to search for a different object. Use when user says they want to find something else or start a new search."""
        logger.info("Function tool 'search_new_object' called - resetting and transferring to Greeting")
        # Reset userdata for new search
        context.userdata.object_to_find = None
        context.userdata.user_location = None
        context.userdata.object_found = False
        return await self._transfer_to_agent("greeter", context)
    
    @function_tool()
    async def end_session(self, context: RunContext_T) -> str:
        """Call this when user wants to end the session or says goodbye, thank you, that's all, I'm done, or similar."""
        logger.info("Function tool 'end_session' called - ending navigation")
        return "Navigation complete. Have a great day!"


async def entrypoint(ctx: agents.JobContext):

    userdata = UserData()

    cwd = os.getcwd()
    filepath = os.path.join(cwd, "data/test.jpg") # in real word case this image will be taken from somewhere like S3 
    userdata.object_image = filepath

    userdata.agents.update(
        {
            "greeter": Greeting(),
            "object_detection": ObjectDetectionAgent(),
            "rag": RAGAgent(),
            "depth_estimation": DepthEstimationAgent(),
        }
    )

    session = AgentSession(
        stt = sarvam.STT(
            model="saaras:v3",
            language=userdata.preferred_language_code
        ),
        llm="google/gemini-3-flash",
        tts=sarvam.TTS(
            target_language_code=userdata.preferred_language_code,
            speaker="shubh",
            model="bulbul:v3"
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={
            "mode": "fixed",
            "min_delay": 0.5,
            "max_delay": 3.0,
            },
            interruption={
            "mode": "adaptive",
            "min_duration": 0.5,
            "resume_false_interruption": True,
            },
        ),
        userdata=userdata
    )

    await session.start(
        room=ctx.room,
        agent=userdata.agents["greeter"],
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# ----------------------------
# Minimal health server
# ----------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server():
    http_server = HTTPServer(("0.0.0.0", 4000), HealthHandler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()

if __name__ == "__main__":  
    # Start lightweight health server
    start_health_server()

    # Boot LiveKit agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))