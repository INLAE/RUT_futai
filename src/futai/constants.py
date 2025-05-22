"""Все строковые и числовые MaGiC литералы в одном месте."""

#  Detection settings
# Порог отсечения детекций по уверенности
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3

# ================================
#  SigLIP + clustering
# Ппуть до предобученной SigLIP-модели в HF Hub
SIGLIP_MODEL_NAME: str = 'google/siglip-base-patch16-224'
# Количество компонент для UMAP
UMAP_N_COMPONENTS: int = 3
# Количество кластеров (команд)
KMEANS_N_CLUSTERS: int = 2
# Девайс по умолчанию
DEFAULT_DEVICE: str = 'cuda'
# Размер батча для извлечения эмбеддингов
DEFAULT_BATCH_SIZE: int = 32

# ================================
#  Class IDs
BALL_CLASS_ID: int       = 0
GK_CLASS_ID: int         = 1
PLAYER_CLASS_ID: int     = 2
REFEREE_CLASS_ID: int    = 3

# ================================
#  Annotation settings
# Палитра HEX кодов цветов для двух команд + дополнительный акцент
COLOR_PALETTE_HEX: list[str] = ['#00BFFF', '#FF1493', '#FFD700']
# Толщина линий для эллипсов
ELLIPSE_THICKNESS: int = 2
# Параметры треугольника над мячом
TRIANGLE_BASE: int             = 25
TRIANGLE_HEIGHT: int           = 21
TRIANGLE_OUTLINE_THICKNESS: int = 1
# Цвет текста (номер трека)
TEXT_COLOR_HEX: str = '#000000'
# Позиция текста под/над боксом
LABEL_TEXT_POSITION = None     # будем брать sv.Position.BOTTOM_CENTER в коде