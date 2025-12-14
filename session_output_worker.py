"""
EventQueue worker for sequential session output processing.
Processes transcription chunks and sends keyboard output.
"""
import queue
from processing_session import ProcessingSession
from lib.pr_log import pr_err, pr_info, pr_debug


def process_session_output(app, session: ProcessingSession):
    """
    Worker function that processes session chunks sequentially for keyboard output.

    This function is called by the EventQueue worker thread, ensuring that
    only one session outputs to the keyboard at a time.
    """
    if session.has_error:
        app.show_error_notification(session.error_message)
        return

    app.transcription_service.keyboard.prepare_for_session(session)
    app.transcription_service.reset_streaming_state()

    notification_shown = False
    current_retry_count = session.retry_count

    try:
        while not session.chunks_complete.is_set() or not session.chunk_queue.empty():
            if session.retry_count > current_retry_count:
                pr_debug("Retry detected, discarding partial state")
                app.transcription_service.reset_streaming_state()
                current_retry_count = session.retry_count
                continue

            try:
                chunk = session.chunk_queue.get(timeout=0.1)

                # Window activation synchronization on first chunk (just-in-time check)
                # Ensures window check happens when chunks are ready, not at session start
                if not notification_shown:
                    notification_shown = True
                    if app.transcription_service.keyboard.is_session_window_active():
                        session.window_activated.set()
                        if app.config.debug_enabled:
                            pr_debug("Session window active, proceeding with output")
                    else:
                        if session.recording_session.window_id is not None:
                            app.show_window_focus_notification(
                                session.recording_session.window_id,
                                "Click here to return to the original window and continue output"
                            )
                            pr_debug("Waiting for window activation event")
                            session.window_activated.wait()
                            pr_debug("Window activation event received, proceeding with output")
                        else:
                            session.window_activated.set()

                app.transcription_service.process_streaming_chunk(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                pr_err(f"Error processing chunk: {e}")

        app.transcription_service.complete_stream()

        final_text = app.transcription_service._build_current_text()
        if final_text:
            pr_info(f"{final_text}\n")
        else:
            pr_info("")

        if app.config.reset_state_each_response:
            app.transcription_service.reset_all_state()
    finally:
        app.transcription_service.keyboard.cleanup_session()
