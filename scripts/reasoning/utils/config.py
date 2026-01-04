import os

def find_project_root(project_name="trainrl"):
    path = os.path.abspath(__file__)
    while True:
        if os.path.basename(path) == project_name:
            return path
        new_path = os.path.dirname(path)
        if new_path == path:
            raise RuntimeError(f"Project root '{project_name}' not found.")
        path = new_path

REQUIRED_PPE_IDS = {0, 1, 2, 3, 4, 5, 7}

PPE_CLASS = {
    0: 'Boots', 1: 'Ear-protection', 2: 'Glass', 3: 'Glove',
    4: 'Helmet', 5: 'Mask', 6: 'Person', 7: 'Vest'
}

RAW_THREAT_CONFIG = {
    0: [0, 1, 5, 6, 7, 8, 9],
    0.2: [2, 3, 4]
}

THREAT_RANGE = {
    "low": [0, 0.3],
    "medium": [0.3, 0.9],
    "high": [0.9, 10]
}