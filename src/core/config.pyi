class Features:
    us: str
    eu: str
    asia: str
    region: str


class Url:
    qa_true_counts: str
    rerated_video_ids: str
    class_labels_indices: str
    ontology: str
    balanced_train_segments: str
    unbalanced_train_segments: str
    eval_segments: str


class Path:
    qa_true_counts: str
    rerated_video_ids: str
    class_labels_indices: str
    ontology: str
    balanced_train_segments: str
    unbalanced_train_segments: str
    eval_segments: str


class Segment:
    ydl_outtmpl: str
    dir: str
    frames_dir: str
    audios_dir: str
    raw: str
    frame: str
    audio: str


class Label:
    dir: str


class Config:
    features: Features
    url: Url
    path: Path
    segment: Segment
    label: Label


config: Config
