import json
import logging
from datetime import datetime


class JsonHandler(logging.Handler):
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.data = {
            'start_time': datetime.now().isoformat(),
            'episodes': [],
            'configuration': {},
            'logs': []
        }
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name
        }
        self.data['logs'].append(log_entry)
    
    def add_episode(self, episode_data):
        self.data['episodes'].append(episode_data)
    
    def set_config(self, config):
        self.data['configuration'] = config
    
    def save(self):
        self.data['end_time'] = datetime.now().isoformat()
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)