"""
Thread worker for model invocation.
Runs in parallel to invoke transcription models asynchronously.
"""
import queue
from typing import Optional
from processing_session import ProcessingSession
from audio_source import AudioResult, AudioDataResult, AudioTextResult
from lib.pr_log import pr_err, pr_warn, pr_debug
from litellm import exceptions as litellm_exceptions


def invoke_model_for_session(provider, session: ProcessingSession, result: AudioResult):
    """
    Thread worker that invokes model and writes chunks to session queue.

    This function runs in a daemon thread spawned by ProcessingCoordinator.
    It routes the audio result to the appropriate model invocation based on type.
    """
    if not provider:
        session.chunks_complete.set()
        return

    try:
        if isinstance(result, AudioDataResult):
            _invoke_model(provider, session, audio_data=result.audio_data)
        elif isinstance(result, AudioTextResult):
            _invoke_model(provider, session, text_data=result.transcribed_text)
        else:
            pr_err(f"Unsupported audio result type: {type(result)}")
            session.chunks_complete.set()
    except Exception as e:
        pr_err(f"Error in invoke_model_for_session: {e}")
        session.chunks_complete.set()


def _rollback_session_state(session: ProcessingSession):
    """Clear partial streaming state before retry."""
    while not session.chunk_queue.empty():
        try:
            session.chunk_queue.get_nowait()
        except queue.Empty:
            break

    session.chunks_complete.clear()
    session.retry_count += 1


def _invoke_model(provider, session: ProcessingSession, audio_data=None, text_data=None):
    """Invoke model with streaming callback that collects chunks to session queue."""
    max_retries = provider.config.retry_count

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                _rollback_session_state(session)
                pr_debug(f"Retry attempt {attempt}/{max_retries} for session")

            def streaming_callback(chunk_text):
                session.chunk_queue.put(chunk_text)

            provider.transcribe(
                session.context,
                audio_data=audio_data,
                text_data=text_data,
                streaming_callback=streaming_callback,
                final_callback=None
            )

            break

        except litellm_exceptions.InternalServerError as e:
            error_msg = "Internal error encountered"
            session.error_message = error_msg
            pr_err(f"Dictation API error: {error_msg}")
            break

        except Exception as e:
            is_timeout = (
                'timeout' in str(e).lower() or
                'readtimeout' in type(e).__name__.lower() or
                'apitimeouterror' in type(e).__name__.lower()
            )

            if is_timeout and attempt < max_retries:
                pr_warn(f"Timeout on attempt {attempt}/{max_retries}: {e}")
                continue
            else:
                error_msg = str(e)
                session.error_message = error_msg
                pr_err(f"Model invocation error: {error_msg}")
                break

    session.chunks_complete.set()
