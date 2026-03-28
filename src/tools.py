import asyncio
import logging
from livekit.agents import get_job_context, function_tool
from livekit import api

logger = logging.getLogger(__name__)

@function_tool
async def end_call(reason: str = "User requested to end the call", confirmation_message: str = "Ending the call now. Goodbye!"):
    """
    Ends the current call by disconnecting the agent from the room after a short delay, performs cleanup, and deletes the room.
    
    Parameters:
    reason (str): The reason for ending the call (default: "User requested to end the call").
    confirmation_message (str): Message to say before disconnecting (default: "Ending the call now. Goodbye!").
    
    Returns:
    str: Status message indicating success or failure.
    """
    try:
        # Get the current job context
        ctx = get_job_context()
        
        # Log the end call request
        logger.info(f"Ending call: {reason}")
        
        # Get room details before shutdown
        room_name = ctx.room.name
        
        # Wait for 5 seconds as per requirements
        await asyncio.sleep(5)
        
        # Disconnect the agent from the room
        await ctx.shutdown()
        
        # Initialize LiveKit API client using environment variables
        lkapi = api.LiveKitAPI.from_env()
        
        # Delete the room
        await lkapi.room.delete_room(
            api.DeleteRoomRequest(room=room_name)
        )
        
        # Close the API client
        await lkapi.aclose()
        
        return f"Call ended successfully: {confirmation_message}. Room '{room_name}' deleted."
    
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        return f"Failed to end call: {str(e)}"
