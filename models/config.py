import json
import logging

logging.basicConfig(level=logging.INFO) 
handler = logging.FileHandler("config.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(handler)
try:
    from . import settings    
except ImportError:
    def load_settings(json_file,provider_type):
        with open(json_file, "r") as f:
            whole_json=json.load(f)
            try:
                return whole_json[provider_type]
            except KeyError:
                raise ValueError(f"Provider type '{provider_type}' not found in {json_file}")
                logging.exception(f"Provider type '{provider_type}' not found in {json_file}")      
        logging.info("Using load_settings function to load settings from config.json")
    logging.exception("Failed to import settings module. Using load_settings function instead.")          