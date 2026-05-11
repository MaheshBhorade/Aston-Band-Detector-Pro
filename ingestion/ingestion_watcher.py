import time
import logging
from pathlib import Path
from queue import Queue
from threading import Thread

from ingestion.ad_ingest_service import AdIngestService
from ingestion.file_utils import wait_until_file_complete
import sys
from pathlib import Path

from watchdog.events import FileSystemEventHandler
import time
from watchdog.observers import Observer
from threading import Lock

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("ad_ingestion.watcher")
# ==========================================================
# Job Queue + Worker Pool
# ==========================================================

job_queue = Queue()

NUM_WORKERS = 4


# Prevent duplicate ingestion
processing_files = set()

processing_lock = Lock()


def worker(service: AdIngestService):
        # ingest_service = AdIngestService()

        while True:

            # video_file, metadata_file = self.queue.get()
            video_path, json_path = job_queue.get()


            try:

                logger.info(f"Processing job: {video_path.name}")

                # ignored_ad_id = self.service.process_ad(video_file, metadata_file)
                service.process_ad(video_path, json_path)


                # if ignored_ad_id:
                #     logger.info(f"Ad ignored: {ignored_ad_id}")

            except Exception as e:

                logger.error(f"Ingestion worker error: {e}")

            finally:

                with processing_lock:

                    if video_path.name in processing_files:
                        processing_files.remove(video_path.name)

                job_queue.task_done()

            # self.queue.task_done()
# ------------------------------------------------
# Scan Directory
# ------------------------------------------------

class IncomingAdsHandler(FileSystemEventHandler):

    def on_created(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() not in [".mp4", ".mov", ".mkv", ".mxf"]:
            return

        json_path = file_path.with_suffix(".json")

        with processing_lock:

            if file_path.name in processing_files:

                logger.debug(f"Ignoring duplicate event: {file_path.name}")

                return

            processing_files.add(file_path.name)

        logger.info(f"New advertisement detected: {file_path.name}")

        # ------------------------------------------------
        # Wait for video copy to finish
        # ------------------------------------------------

        if not wait_until_file_complete(file_path):

            logger.error(f"File copy incomplete: {file_path}")

            with processing_lock:
                processing_files.remove(file_path.name)

            return

        # ------------------------------------------------
        # Wait for JSON metadata
        # ------------------------------------------------

        wait_time = 0

        while not json_path.exists():

            time.sleep(1)

            wait_time += 1

            if wait_time > 15:

                logger.warning(f"JSON metadata missing for {file_path}")

                with processing_lock:
                    processing_files.remove(file_path.name)

                return

        logger.info(f"Queueing advertisement: {file_path.name}")

        job_queue.put((file_path, json_path))
    
# class IngestionWatcher:

#     def __init__(self, scan_interval=10, workers = 4):



#         self.incoming_dir = Path("storage/incoming_ads")
#         self.scan_interval = scan_interval
#         self.workers = workers

#         self.queue = Queue()

#         # track queued jobs
#         self.seen_jobs = set()

#         self.service = AdIngestService()

#         self.incoming_dir.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------
#     # Scan Directory
#     # ------------------------------------------------

#     def scan_directory(self):

#         video_files = list(self.incoming_dir.glob("*.mp4"))

#         for video_file in video_files:

#             metadata_file = video_file.with_suffix(".json")

#             if not metadata_file.exists():
#                 continue

#             job = (video_file, metadata_file)

#             if job not in list(self.queue.queue):

#                 logger.info(f"Queueing new advertisement: {video_file.name}")

#                 self.queue.put((video_file, metadata_file))
#                 self.seen_jobs.add(job)
def scan_existing_files(incoming_dir: Path):
    logger.info("Scanning existing files on startup...")

    video_extensions = [".mp4", ".mov", ".mkv", ".mxf"]

    for video_path in incoming_dir.iterdir():

        if not video_path.is_file():
            continue

        if video_path.suffix.lower() not in video_extensions:
            continue

        json_path = video_path.with_suffix(".json")

        if not json_path.exists():
            logger.debug(f"Skipping (no metadata yet): {video_path.name}")
            continue

        with processing_lock:
            if video_path.name in processing_files:
                continue
            processing_files.add(video_path.name)

        # Optional: ensure file is fully written (important if crash happened mid-copy)
        if not wait_until_file_complete(video_path):
            logger.warning(f"Skipping incomplete file: {video_path.name}")
            with processing_lock:
                processing_files.remove(video_path.name)
            continue

        logger.info(f"Queueing existing advertisement: {video_path.name}")
        job_queue.put((video_path, json_path))

# ==========================================================
# Watchdog Start
# ==========================================================

def start_watcher():

    incoming_dir = Path("storage/incoming_ads")

    event_handler = IncomingAdsHandler()

    observer = Observer()

    observer.schedule(event_handler, str(incoming_dir), recursive=False)

    observer.start()

    logger.info("Advertisement ingestion watcher started (Watchdog mode)")

    shared_service = AdIngestService()
    # ------------------------------------------------
    # Start workers
    # ------------------------------------------------

    for _ in range(NUM_WORKERS):

        t = Thread(target=worker, args=(shared_service,), daemon=True)

        t.start()
        
    scan_existing_files(incoming_dir)
    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        observer.stop()

    observer.join()

    

    # ------------------------------------------------
    # Start Watcher
    # ------------------------------------------------

    # def start(self):

    #     logger.info("Advertisement ingestion watcher started")

    #     for _ in range(self.workers):

    #         thread = Thread(target=self.worker, daemon=True)
    #         thread.start()

    #     while True:

    #         try:

    #             self.scan_directory()

    #         except Exception as e:

    #             logger.error(f"Watcher error: {e}")

    #         time.sleep(self.scan_interval)


#     def start(self):

#         logger.info("Advertisement ingestion watcher started")

#         for _ in range(self.workers):
#             thread = Thread(target=self.worker, daemon=True)
#             thread.start()

#         self.scan_directory()

#         handler = IncomingAdsHandler(self)

#         observer = Observer()
#         observer.schedule(handler, str(self.incoming_dir), recursive=False)
#         observer.start()

#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             observer.stop()

#         observer.join()

# if __name__ == "__main__":

#     watcher = IngestionWatcher(
#         scan_interval=10,
#         workers=2
#     )

#     watcher.start()
if __name__ == "__main__":

    start_watcher()