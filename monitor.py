import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from logger_setup import logger

class ProjectMonitorHandler(FileSystemEventHandler):
    def on_created(self, event):
        # SKIP git and pycache folders
        if ".git" in event.src_path or "__pycache__" in event.src_path:
            return
            
        item_type = "Folder" if event.is_directory else "File"
        logger.info(f"NEW {item_type} CREATED: {event.src_path}")

    def on_moved(self, event):
        if ".git" in event.src_path or "__pycache__" in event.src_path:
            return
        logger.info(f"MOVED/RENAMED: {event.src_path} TO {event.dest_path}")

if __name__ == "__main__":
    path = "." 
    event_handler = ProjectMonitorHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    
    logger.info(f"Started CLEAN monitoring (ignoring .git and pycache)")
    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()