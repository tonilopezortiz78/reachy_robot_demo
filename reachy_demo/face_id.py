"""
reachy_demo/face_id.py — face identification for the unified demo.

Drop-in upgrade from the classic demo_face_recognition.py dlib pipeline.

Stack (all Apache-2.0 / MIT, ~37 MB total):
    YuNet (320 KB, ~3 ms @ 640×360)        — face detector + 5 landmarks
    SFace (36 MB, ~5 ms int8)              — 128-D embedding via ONNX Runtime CPU
    IoU tracker (~200 LOC)                 — identity persistence between detections

Roster UX preserved: drop photos in faces/<name>/, restart — no retrain.

Models download on first use to cache/models/:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx

If models can't be fetched (offline), this module gracefully falls back to
the already-installed legacy `face_recognition` dlib package — same API,
slower, same accuracy, but the demo doesn't crash.

Public API:
    fid = FaceIdentifier(faces_dir, cache_dir)
    roster_count = fid.load_roster()
    boxes = fid.detect(rgb_frame)
    results = fid.identify(rgb_frame) -> [(box, name, conf, track_id)]
    fid.identify_typical_box_count_in_box_count -> heuristic ≈ 1
"""

import shutil
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Model URLs ────────────────────────────────────────────────────────────────

_MODELS = {
    "yunet": {
        "url":  "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "fname": "face_detection_yunet_2023mar.onnx",
        "md":    None,
    },
    "sface": {
        "url":  "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "fname": "face_recognition_sface_2021dec.onnx",
        "md":    None,
    },
}

# SFace alignment template (5-point) — taken from the OpenCV zoo demo.
_ARC_FACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Cosine-distance acceptance threshold (~0.40 → ~99% same-person on SFace LFW).
DEFAULT_TOLERANCE = 0.40
RECOG_INTERVAL = 15   # frames between recognition runs per track
# Minimum face box size (px, in the ~640-wide capture frame) to attempt a match.
# Small = far away = low-res crop → SFace embeddings become unreliable and get
# force-matched to whoever is enrolled (the "everyone is Tony" bug). Below this,
# treat the person as an unidentified visitor until they step closer.
MIN_RECOG_PX = 84
ENROLL_IOU_MIN = 0.15  # min IoU vs target_box for targeted enrolment (person may shift)

RELATION_NORM = np.linalg.norm(_ARC_FACE_REF, axis=None) or 1.0


def _download(url: str, dest: Path, timeout: float = 30.0) -> bool:
    """Download with timeout; print progress. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [face-id] downloading {dest.name}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reachy-mini/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            dest.write_bytes(r.read())
        print(f"  [face-id] {dest.name} ({dest.stat().st_size/1e6:.1f} MB) ok")
        return True
    except Exception as e:
        print(f"  [face-id] download failed: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def _align_faces(frame_bgr: np.ndarray, landmarks_per_face: list) -> list[np.ndarray]:
    """Affine-warp each face to a 112×112 canonical pose via 5 landmarks.
    Returns a list of aligned (112,112,3) BGR uint8 crops.
    Pure-numpy implementation (no scipy dep on the critical path)."""
    out = []
    sx = 112 / 112.0
    sy = 112 / 112.0
    for lm in landmarks_per_face:
        lm = np.asarray(lm, dtype=np.float32).reshape(5, 2)
        ref = _ARC_FACE_REF * np.array([sx, sy], dtype=np.float32)
        M = cv2.estimateAffinePartial2D(lm, ref, method=cv2.LMEDS)[0]
        if M is None:
            continue
        out.append(cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0))
    return out


def _iou(box_a: tuple, box_b: tuple) -> float:
    """IoU of two (x1,y1,x2,y2) boxes — same math as _IoUTracker.update."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * \
            max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + \
            (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / n)


class _IoUTracker:
    """Tiny IoU-only face tracker — assigns a stable track_id per face across
    frames so recognition only runs at track birth + every RECOG_INTERVAL
    frames. Single-face, close-range robot scenario: faces are sparse and slow,
    so naive IoU matching + reuse_last_id covers it."""

    def __init__(self, iou_thresh: float = 0.30):
        self.iou_thresh = iou_thresh
        self._next_id = 0
        self._tracks: dict[int, list] = {}  # track_id -> [x1,y1,x2,y2]
        self._frames_since_recog: dict[int, int] = {}

    def update(self, detections: list[tuple[int, int, int, int]]) -> list[tuple[tuple, int]]:
        """Each detection -> ((x1,y1,x2,y2), track_id)."""
        matched: list[tuple[tuple, int]] = []
        unmatched_tracks = set(self._tracks.keys())
        for det in detections:
            bx1, by1, bx2, by2 = det
            best_id, best_iou = None, self.iou_thresh
            for tid in list(unmatched_tracks):
                tx1, ty1, tx2, ty2 = self._tracks[tid]
                inter = max(0, min(bx2, tx2) - max(bx1, tx1)) * \
                        max(0, min(by2, ty2) - max(by1, ty1))
                union = (bx2 - bx1) * (by2 - by1) + \
                        (tx2 - tx1) * (ty2 - ty1) - inter
                iou = inter / union if union > 0 else 0
                if iou >= best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._frames_since_recog[best_id] = RECOG_INTERVAL
            self._tracks[best_id] = [bx1, by1, bx2, by2]
            self._frames_since_recog[best_id] += 1
            unmatched_tracks.discard(best_id)
            matched.append(((bx1, by1, bx2, by2), best_id))
        # Prune dead tracks
        for tid in list(self._tracks.keys()):
            if tid not in [m[1] for m in matched]:
                del self._tracks[tid]
                self._frames_since_recog.pop(tid, None)
        return matched

    def needs_recog(self, tid: int) -> bool:
        n = self._frames_since_recog.get(tid, RECOG_INTERVAL)
        return n >= RECOG_INTERVAL

    def reset_recog(self, tid: int):
        self._frames_since_recog[tid] = 0


def _alignment_landmarks(face_landmarks: np.ndarray) -> np.ndarray:
    """YuNet returns [[x,y], ...] for [right_eye, left_eye, nose,
    right_mouth, left_mouth]. SFace's expected order is
    [left_eye, right_eye, nose, right_mouth, left_mouth]."""
    if face_landmarks is None or len(face_landmarks) == 0:
        return np.zeros((5, 2), dtype=np.float32)
    lm = np.asarray(face_landmarks, dtype=np.float32).reshape(-1, 2)
    if lm.shape[0] < 5:
        pad = np.zeros((5 - lm.shape[0], 2), dtype=np.float32)
        lm = np.vstack([lm, pad])
    # Swap right/left eye indices
    return lm[[1, 0, 2, 3, 4]]


class FaceIdentifier:
    """YuNet + SFace + IoU tracker + folder roster."""

    def __init__(self, faces_dir: Path, cache_dir: Path,
                 tol: float = DEFAULT_TOLERANCE, mirror: bool = False):
        self.faces_dir = Path(faces_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tol = tol
        self.mirror = mirror
        self._lock = threading.Lock()
        self._yunet: cv2.FaceDetectorYN | None = None
        self._sface: cv2.dnn.Net | None = None
        self._ref_embs: np.ndarray | None = None
        self._ref_names: list[str] = []
        self._tracker = _IoUTracker()
        self._use_modern = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def init_models(self) -> bool:
        """Try to download + load YuNet+SFace. Returns False → caller falls
        back to legacy face_recognition (loaded separately)."""
        ypath = self.cache_dir / _MODELS["yunet"]["fname"]
        spath = self.cache_dir / _MODELS["sface"]["fname"]
        ok_y = _download(_MODELS["yunet"]["url"], ypath)
        ok_s = _download(_MODELS["sface"]["url"], spath)
        if not (ok_y and ok_s):
            return False
        try:
            self._yunet = cv2.FaceDetectorYN.create(
                str(ypath), "", (640, 360),
                score_threshold=0.6, nms_threshold=0.3, top_k=10,
            )
            self._sface = cv2.dnn.readNetFromONNX(str(spath))
            self._use_modern = True
            return True
        except Exception as e:
            print(f"  [face-id] ONNX load failed: {e}")
            return False

    @property
    def using_modern(self) -> bool:
        return self._use_modern

    # ── Roster ────────────────────────────────────────────────────────────────

    def load_roster(self) -> int:
        """Encode faces/<name>/*.jpg into self._ref_embs. Returns # people."""
        if self._use_modern:
            return self._load_roster_sface()
        return self._load_roster_dlib()

    def _roster_photos(self) -> list[tuple[str, str]]:
        photos = []
        if not self.faces_dir.exists():
            return photos
        for person_dir in sorted(self.faces_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue
            name = person_dir.name.replace("_", " ").title()
            for p in sorted(person_dir.glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    photos.append((name, str(p)))
        return photos

    def _encode_sface(self, bgr_crop_112: np.ndarray) -> np.ndarray | None:
        blob = cv2.dnn.blobFromImage(
            bgr_crop_112, 1.0 / 127.5, (112, 112),
            (127.5, 127.5, 127.5), swapRB=False)
        self._sface.setInput(blob)
        emb = self._sface.forward().flatten().astype(np.float32)
        return emb

    def _load_roster_sface(self) -> int:
        photos = self._roster_photos()
        if not photos:
            print("  [roster] no photos — everyone will be a visitor")
            self._ref_embs = np.zeros((0, 128), dtype=np.float32)
            self._ref_names = []
            return 0
        embs, names = [], []
        peoples: set[str] = set()
        ydet = self._yunet
        for name, path in photos:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            ydet.setInputSize((w, h))
            _, faces = ydet.detect(img)
            if faces is None or len(faces) == 0:
                print(f"  [roster] {Path(path).name}: no face — skip")
                continue
            lm = _alignment_landmarks(faces[0, 4:14].reshape(5, 2))
            aligned = _align_faces(img, [lm])
            if not aligned:
                continue
            emb = self._encode_sface(aligned[0])
            if emb is None:
                continue
            embs.append(emb)
            names.append(name)
            peoples.add(name)
        self._ref_embs = np.asarray(embs, dtype=np.float32)
        self._ref_names = names
        print(f"  [roster] {len(peoples)} people, {len(embs)} encodings (SFace)")
        return len(peoples)

    def _load_roster_dlib(self) -> int:
        """Legacy path using the existing face_recognition / dlib install."""
        try:
            import face_recognition
        except ImportError:
            print("  [roster] no face_recognition package — empty roster")
            self._ref_embs = None
            self._ref_names = []
            return 0
        photos = self._roster_photos()
        peoples: set[str] = set()
        embs, names = [], []
        for name, path in photos:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                continue
            embs.append(encs[0])
            names.append(name)
            peoples.add(name)
        self._ref_embs = np.asarray(embs, dtype=np.float32) if embs else None
        self._ref_names = names
        print(f"  [roster] {len(peoples)} people, {len(embs)} encodings (dlib fallback)")
        return len(peoples)

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, rgb_frame: np.ndarray) -> list:
        """Return YuNet detections: list of [box(4), score, 5 landmarks]."""
        if self._use_modern:
            return self._detect_yunet(rgb_frame)
        return self._detect_dlib(rgb_frame)

    def _detect_yunet(self, rgb_frame: np.ndarray) -> list:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if (w, h) != tuple(self._yunet.getInputSize()):
            self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(bgr)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, w_box, h_box = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score = float(f[2 + 5 + 5]) if len(f) > 14 else 0.0
            lm = f[4:14].reshape(5, 2)
            out.append(((x, y, x + w_box, y + h_box), score, lm))
        return out

    def _detect_dlib(self, rgb_frame: np.ndarray) -> list:
        import face_recognition
        small = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        locs = face_recognition.face_locations(small, model="hog")
        inv = 1.0 / 0.5
        out = []
        for (top, right, bottom, left) in locs:
            box = (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
            out.append((box, 1.0, None))
        return out

    # ── Identification ─────────────────────────────────────────────────────────

    def identify(self, rgb_frame: np.ndarray) -> list[tuple[tuple, str, float, int]]:
        """Full pipeline: detect → track → recognize at track birth + every N
        frames. Returns [(box(x1,y1,x2,y2), name, conf, track_id)]."""
        with self._lock:
            if self.mirror:
                work = cv2.flip(rgb_frame, 1)
            else:
                work = rgb_frame
            dets = self._yk_det(work) if self._use_modern else \
                self._dl_det(work)
            det_boxes = [d[0] for d in dets]
            tracked = self._tracker.update(det_boxes)
            out = []
            for (box, tid) in tracked:
                name, conf = "visitor", 0.0
                if self._tracker.needs_recog(tid):
                    name, conf = self._recognize_box(work, box, dets)
                    self._tracker.reset_recog(tid)
                else:
                    cached = getattr(self, "_cache", {}).get(tid)
                    if cached:
                        name, conf = cached
                cache = getattr(self, "_cache", {})
                cache[tid] = (name, conf)
                self._cache = cache
                out.append((box, name, conf, tid))
            if self.mirror:
                W = rgb_frame.shape[1]
                out = [((W - b[2], b[1], W - b[0], b[3]), n, c, t)
                       for (b, n, c, t) in out]
            return out

    def add_person(self, name: str, rgb_frames: list) -> int:
        """Live-add a new person to the roster without restarting.

        Detects the largest face in each frame, encodes it via SFace (or dlib
        fallback), saves the cropped face images to faces/<name>/, and appends
        the embeddings to the in-memory roster. Returns count of usable encodings.
        Thread-safe via self._lock.
        """
        with self._lock:
            pname = name.lower().replace(" ", "_")
            pdir = self.faces_dir / pname
            pdir.mkdir(parents=True, exist_ok=True)
            new_embs = []
            added = 0
            for rgb_frame in rgb_frames:
                ts = int(time.time() * 1000)
                fname = pdir / f"{pname}_{ts}_{added}.jpg"
                if self._use_modern:
                    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    if (w, h) != tuple(self._yunet.getInputSize()):
                        self._yunet.setInputSize((w, h))
                    _, faces = self._yunet.detect(bgr)
                    if faces is None or len(faces) == 0:
                        continue
                    largest = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                    lm = _alignment_landmarks(largest[4:14].reshape(5, 2))
                    aligned = _align_faces(bgr, [lm])
                    if not aligned:
                        continue
                    emb = self._encode_sface(aligned[0])
                    if emb is None:
                        continue
                    cv2.imwrite(str(fname), bgr)
                    new_embs.append(emb)
                    added += 1
                else:
                    try:
                        import face_recognition
                    except ImportError:
                        break
                    encs = face_recognition.face_encodings(rgb_frame)
                    if not encs:
                        continue
                    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(fname), bgr)
                    new_embs.append(encs[0])
                    added += 1
            if new_embs:
                stacked = np.asarray(new_embs, dtype=np.float32)
                if self._ref_embs is not None and len(self._ref_embs) > 0:
                    self._ref_embs = np.vstack([self._ref_embs, stacked])
                else:
                    self._ref_embs = stacked
                self._ref_names.extend([name] * added)
            return added

    def add_person_targeted(self, name: str, rgb_frames: list,
                            target_box=None, exclude_known: bool = True) -> int:
        """Like add_person, but enrols the RIGHT face when several are in view.

        Per frame, candidate faces are ranked:
          - target_box given ((x1,y1,x2,y2) in identify() pixel coords) —
            best IoU with target_box first, requiring IoU >= ENROLL_IOU_MIN;
            if no face passes the bar, fall back to largest-first.
          - target_box None — largest-first (add_person behaviour).
        With exclude_known=True each candidate is encoded and checked against
        the existing roster (same measure/threshold as _recognize_box); a face
        that matches a known person is rejected and the next candidate tried —
        so a bystander like Tony can never be re-enrolled under a new name.

        dlib fallback (documented simplification): full-res HOG detect, boxes
        converted to (x1,y1,x2,y2), same IoU/largest ranking, roster check via
        face_distance < 0.52 as in _recognize_box.

        Saves accepted frames to faces/<slug>/ and appends embeddings to the
        in-memory roster like add_person. The folder is created lazily, so a
        fully-rejected enrolment leaves no empty directory. Returns count of
        usable encodings. Thread-safe via self._lock.
        """
        with self._lock:
            pname = name.lower().replace(" ", "_")
            pdir = self.faces_dir / pname
            new_embs = []
            added = 0
            for rgb_frame in rgb_frames:
                ts = int(time.time() * 1000)
                fname = pdir / f"{pname}_{ts}_{added}.jpg"
                if self._use_modern:
                    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    if (w, h) != tuple(self._yunet.getInputSize()):
                        self._yunet.setInputSize((w, h))
                    _, faces = self._yunet.detect(bgr)
                    if faces is None or len(faces) == 0:
                        continue
                    cands = [((int(f[0]), int(f[1]),
                               int(f[0] + f[2]), int(f[1] + f[3])),
                              f[4:14].reshape(5, 2)) for f in faces]
                    ranked = self._rank_candidates(cands, target_box)
                    for box, raw_lm in ranked:
                        lm = _alignment_landmarks(raw_lm)
                        aligned = _align_faces(bgr, [lm])
                        if not aligned:
                            continue
                        emb = self._encode_sface(aligned[0])
                        if emb is None:
                            continue
                        if exclude_known and self._matches_roster(emb):
                            continue  # known person — try next candidate
                        pdir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(fname), bgr)
                        new_embs.append(emb)
                        added += 1
                        break
                else:
                    try:
                        import face_recognition
                    except ImportError:
                        break
                    locs = face_recognition.face_locations(rgb_frame, model="hog")
                    if not locs:
                        continue
                    # (top,right,bottom,left) -> (x1,y1,x2,y2)
                    cands = [((loc[3], loc[0], loc[1], loc[2]), loc)
                             for loc in locs]
                    ranked = self._rank_candidates(cands, target_box)
                    accepted = None
                    for box, loc in ranked:
                        encs = face_recognition.face_encodings(rgb_frame, [loc])
                        if not encs:
                            continue
                        if exclude_known and self._matches_roster(encs[0]):
                            continue
                        accepted = encs[0]
                        break
                    if accepted is None:
                        continue
                    pdir.mkdir(parents=True, exist_ok=True)
                    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(fname), bgr)
                    new_embs.append(accepted)
                    added += 1
            if new_embs:
                stacked = np.asarray(new_embs, dtype=np.float32)
                if self._ref_embs is not None and len(self._ref_embs) > 0:
                    self._ref_embs = np.vstack([self._ref_embs, stacked])
                else:
                    self._ref_embs = stacked
                self._ref_names.extend([name] * added)
            return added

    @staticmethod
    def _rank_candidates(cands: list, target_box) -> list:
        """Order (box, payload) candidates for targeted enrolment: best IoU
        with target_box first (>= ENROLL_IOU_MIN only), falling back to
        largest-area-first if no target or nothing passes the bar."""
        if target_box is not None:
            scored = [(c, _iou(c[0], target_box)) for c in cands]
            hits = [c for c, s in sorted(scored, key=lambda p: p[1], reverse=True)
                    if s >= ENROLL_IOU_MIN]
            if hits:
                return hits
        return sorted(cands, key=lambda c: (c[0][2] - c[0][0]) *
                      (c[0][3] - c[0][1]), reverse=True)

    def _matches_roster(self, emb: np.ndarray) -> bool:
        """True if emb matches an already-enrolled person, using the same
        measure/threshold as _recognize_box (cosine > self.tol for SFace,
        face_distance < 0.52 for the dlib fallback)."""
        if self._ref_embs is None or len(self._ref_embs) == 0:
            return False
        if self._use_modern:
            sims = np.array([_cosine(emb, ref) for ref in self._ref_embs])
            return float(np.max(sims)) > self.tol
        import face_recognition
        dists = face_recognition.face_distance(self._ref_embs, emb)
        return float(np.min(dists)) < 0.52

    def remove_person(self, name: str) -> int:
        """Remove a person from the roster (in-memory + on-disk).

        Used to correct a mis-saved name (e.g. the visitor says "my name is
        X" but a previous onboarding stored a bad name). Removes every
        roster entry whose stored name matches `name` case-insensitively,
        deletes the on-disk faces/<slug>/ directory, and resets the tracker
        + recognition cache so a just-renamed face is re-recognized fresh
        rather than served the stale cached name.

        Returns the number of embeddings removed. Thread-safe via self._lock.
        """
        with self._lock:
            target = name.strip().lower()
            slug = target.replace(" ", "_")

            keep_idx = [
                i for i, n in enumerate(self._ref_names)
                if n.strip().lower() != target
            ]
            removed = len(self._ref_names) - len(keep_idx)

            if keep_idx:
                if self._ref_embs is not None:
                    self._ref_embs = self._ref_embs[keep_idx]
                self._ref_names = [self._ref_names[i] for i in keep_idx]
            else:
                self._ref_embs = np.zeros((0, 128), dtype=np.float32)
                self._ref_names = []

            pdir = self.faces_dir / slug
            if pdir.exists():
                shutil.rmtree(pdir, ignore_errors=True)

            self._tracker = _IoUTracker()
            self._cache = {}

            return removed

    def _yk_det(self, rgb_frame: np.ndarray):
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if (w, h) != tuple(self._yunet.getInputSize()):
            self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(bgr)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, w_box, h_box = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            lm = f[4:14].reshape(5, 2)
            out.append(((x, y, x + w_box, y + h_box), float(f[14]) if len(f) > 14 else 1.0, lm))
        return out

    def _dl_det(self, rgb_frame: np.ndarray):
        import face_recognition
        small = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        locs = face_recognition.face_locations(small, model="hog")
        inv = 1.0 / 0.5
        out = []
        for (top, right, bottom, left) in locs:
            out.append(((int(top * inv), int(right * inv),
                         int(bottom * inv), int(left * inv)), 1.0, None))
        return out

    def _recognize_box(self, rgb_frame: np.ndarray,
                       box: tuple, dets: list) -> tuple[str, float]:
        if self._ref_embs is None or len(self._ref_embs) == 0:
            return ("visitor", 0.0)
        # Too-small face = too far for a trustworthy embedding → don't guess a name.
        if (box[2] - box[0]) < MIN_RECOG_PX or (box[3] - box[1]) < MIN_RECOG_PX:
            return ("visitor", 0.0)
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        if self._use_modern:
            # Find landmarks for this box
            lm = None
            for (dbox, score, dlm) in dets:
                if dbox == box and dlm is not None:
                    lm = _alignment_landmarks(dlm)
                    break
            if lm is None:
                return ("visitor", 0.0)
            aligned = _align_faces(bgr, [lm])
            if not aligned:
                return ("visitor", 0.0)
            emb = self._encode_sface(aligned[0])
            if emb is None:
                return ("visitor", 0.0)
            sims = np.array([_cosine(emb, ref) for ref in self._ref_embs])
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim > self.tol:
                return (self._ref_names[best_idx], best_sim)
            return ("visitor", best_sim)
        else:
            import face_recognition
            x1, y1, x2, y2 = box
            top, right, bottom, left = y1, x2, y2, x1
            small_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            inv = 1.0 / 0.5
            encs = face_recognition.face_encodings(small_rgb,
                [(int(top * inv * 0.5), int(right * inv * 0.5),
                  int(bottom * inv * 0.5), int(left * inv * 0.5))])
            if not encs:
                return ("visitor", 0.0)
            emb = encs[0]
            dists = face_recognition.face_distance(self._ref_embs, emb)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist < 0.52:
                return (self._ref_names[best_idx], 1.0 - best_dist)
            return ("visitor", 1.0 - best_dist)

    def reset(self):
        self._tracker = _IoUTracker()
        self._cache = {}