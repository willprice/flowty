from pathlib import Path


MEDIA_ROOT = Path(__file__).parent.parent / 'media'
VIDEO_PATHS = {
    'mp4': str(MEDIA_ROOT / 'mr-bubz.mp4'),
    'webm': str(MEDIA_ROOT / 'mr-bubz.webm'),
    'avi': str(MEDIA_ROOT / 'mr-bubz.avi'),
    'jpeg': str(MEDIA_ROOT / 'mr-bubz' / 'frame_%05d.jpg'),
}
RUBBER_WHALE = {
    'media_path': {
        'mp4': str(MEDIA_ROOT / 'rubber-whale.mp4'),
        'png': str(MEDIA_ROOT / 'rubber-whale' / 'frame%02d.png'),
    },
    'resolution': (388, 584),
    'frame_count': 8,
}