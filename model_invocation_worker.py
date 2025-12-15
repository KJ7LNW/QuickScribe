"""
Thread worker for model invocation.
Runs in parallel to invoke transcription models asynchronously.
"""
import queue
import threading
import time
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


def _transcription_worker(provider, context, audio_data, text_data, activity_queue):
    """
    Worker thread that executes provider.transcribe() and signals activity.

    Runs in daemon thread spawned per retry attempt.
    Writes chunks to activity_queue for monitoring thread to consume.
    """
    def streaming_callback(chunk_text):
        activity_queue.put(chunk_text)

    try:
        provider.transcribe(
            context,
            audio_data=audio_data,
            text_data=text_data,
            streaming_callback=streaming_callback,
            final_callback=None
        )
        activity_queue.put(('done', None))

    except Exception as e:
        activity_queue.put(('error', e))


def _invoke_model(provider, session: ProcessingSession, audio_data=None, text_data=None):
    """
    Invoke model with thread-based monitoring for select-timeout capability.

    Spawns worker thread per attempt and monitors activity_queue with timeout.
    """
    max_retries = provider.config.retry_count
    timeout_threshold = provider.config.chunk_timeout
    transcription_succeeded = False

    for attempt in range(1, max_retries + 1):
        if transcription_succeeded:
            break

        if attempt > 1:
            _rollback_session_state(session)
            pr_debug(f"Retry attempt {attempt}/{max_retries} for session")

        # Worker spawning
        activity_queue = queue.Queue()

        worker = threading.Thread(
            target=_transcription_worker,
            args=(provider, session.context, audio_data, text_data, activity_queue),
            daemon=True
        )
        worker.start()

        # Monitoring loop with select timeout
        try:
            last_activity = time.time()
            worker_completed = False

            while not worker_completed:
                remaining = timeout_threshold - (time.time() - last_activity)
                if remaining <= 0:
                    elapsed = time.time() - last_activity
                    raise TimeoutError(f"No chunk for {elapsed:.1f}s")

                try:
                    item = activity_queue.get(timeout=remaining)
                except queue.Empty:
                    elapsed = time.time() - last_activity
                    raise TimeoutError(f"No chunk for {elapsed:.1f}s")

                if isinstance(item, tuple):
                    if item[0] == 'done':
                        pr_debug("Transcription worker completed")
                        worker_completed = True
                    elif item[0] == 'keepalive':
                        last_activity = time.time()
                        pr_debug("Keepalive received, timeout timer reset")
                    elif item[0] == 'error':
                        raise item[1]
                    else:
                        raise RuntimeError(f"Unknown activity_queue signal: {item}")
                else:
                    session.chunk_queue.put(item)
                    last_activity = time.time()
                    pr_debug("Chunk received, timeout timer reset")

            transcription_succeeded = True

        except TimeoutError as e:
            pr_warn(f"Timeout on attempt {attempt}/{max_retries}: {e}")
            if attempt >= max_retries:
                session.error_message = f"Timeout after {max_retries} attempts"
                pr_err(f"Max retries ({max_retries}) exhausted")

        except litellm_exceptions.InternalServerError as e:
            session.error_message = "Internal error encountered"
            pr_err(f"Dictation API error: {session.error_message}")
            break

        except Exception as e:
            is_timeout = (
                'timeout' in str(e).lower() or
                'readtimeout' in type(e).__name__.lower() or
                'apitimeouterror' in type(e).__name__.lower()
            )

            if is_timeout and attempt < max_retries:
                pr_warn(f"Timeout on attempt {attempt}/{max_retries}: {e}")
            else:
                session.error_message = str(e)
                pr_err(f"Model invocation error: {session.error_message}")
                break

    session.chunks_complete.set()
