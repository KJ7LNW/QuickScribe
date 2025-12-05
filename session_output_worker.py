"""
EventQueue worker for sequential session output processing.
Processes transcription chunks and sends keyboard output.
"""
import queue
from processing_session import ProcessingSession
from lib.pr_log import pr_err, pr_info


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
    try:
        while not session.chunks_complete.is_set() or not session.chunk_queue.empty():
            try:
                chunk = session.chunk_queue.get(timeout=0.1)

                if not notification_shown:
                    notification_shown = True
                    if not app.transcription_service.keyboard.is_session_window_active():
                        if session.recording_session.window_id is not None:
                            app.show_window_focus_notification(
                                session.recording_session.window_id,
                                "Click here to return to the original window and continue output"
                            )

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
