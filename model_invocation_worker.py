"""
Thread worker for model invocation.
Runs in parallel to invoke transcription models asynchronously.
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from processing_session import ProcessingSession
from audio_source import AudioResult, AudioDataResult, AudioTextResult
from transcription.types import TranscriptionInput
from lib.pr_log import pr_err, pr_warn, pr_debug
from litellm import exceptions as litellm_exceptions


def _is_transcription_insufficient(text: Optional[str]) -> bool:
    """Check if transcription text is insufficient for model invocation."""
    return text is None or text == '<none>' or text.strip() == ''


def invoke_model_for_session(provider, transcription_source, session: ProcessingSession, results: list[AudioResult]):
    """
    Thread worker that invokes model and writes chunks to session queue.

    This function runs in a daemon thread spawned by ProcessingCoordinator.
    Performs deferred transcription for transcription mode, then routes
    to appropriate model invocation based on result type.
    """
    if not provider:
        session.chunks_complete.set()
        return

    try:
        # Dispatch logic: transcription mode vs raw mode
        # Transcription mode: each audio variant is transcribed separately,
        #   results are wrapped with speed-tagged TX elements and combined
        # Raw mode: all audio variants passed directly to provider (for multimodal models)
        if provider.config.is_transcription_mode():
            transcriptions = []
            seen_texts = set()

            # Parallel transcription of speed variants
            inputs = [
                TranscriptionInput(r.audio_data, r.sample_rate, r.speed_pct)
                for r in results if isinstance(r, AudioDataResult)
            ]

            if inputs:
                with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
                    session.transcription_results = list(executor.map(
                        transcription_source.transcribe_audio_data, inputs))

            # Handle pre-transcribed text results
            for result in results:
                if isinstance(result, AudioTextResult):
                    if not _is_transcription_insufficient(result.transcribed_text):
                        if result.transcribed_text not in seen_texts:
                            seen_texts.add(result.transcribed_text)
                            speed_pct = getattr(result, 'speed_pct', 100)
                            transcriptions.append(f'<tx speed="{speed_pct}%">{result.transcribed_text}</tx>')

            # Deduplication after parallel collection
            for tr_result in session.transcription_results:
                if tr_result.error is None and not _is_transcription_insufficient(tr_result.text):
                    if tr_result.text not in seen_texts:
                        seen_texts.add(tr_result.text)
                        transcriptions.append(f'<tx speed="{tr_result.speed_pct}%">{tr_result.text}</tx>')

            if not transcriptions:
                pr_warn("Skipping model invocation: insufficient transcription from audio")
                session.chunks_complete.set()
                return

            combined_text = '\n'.join(transcriptions)
            _invoke_model(provider, session, text_data=combined_text)

        # Raw mode: pass all audio variants
        else:
            audio_arrays = [r.audio_data for r in results if isinstance(r, AudioDataResult)]
            if not audio_arrays:
                pr_warn("Skipping model invocation: no audio data in results")
                session.chunks_complete.set()
                return

            _invoke_model(provider, session, audio_data=audio_arrays)

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
