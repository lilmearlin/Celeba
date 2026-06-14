import base64
import os
import threading

import cv2
import flet as ft
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

CELEBA_IMAGE_DIR = r"C:\celeba_\img_align_celeba"
CELEBA_ROOT = r"C:\celeba_"
TOP_N = 3
CANDIDATE_POOL = 200

DISTANCE_SAME_PERSON = 1.0
DISTANCE_VERY_CLOSE = 0.6

COLORS = {
    "bg": "#080C18",
    "surface": "#111827",
    "surface_light": "#1A2235",
    "border": "#2A3550",
    "purple": "#7C5CFF",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "gold": "#FBBF24",
    "silver": "#94A3B8",
    "bronze": "#D97706",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "danger": "#F87171",
}


def find_celeba_file(filename):
    for path in (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        filename,
        os.path.join(CELEBA_ROOT, filename),
        os.path.join(os.path.dirname(CELEBA_IMAGE_DIR), filename),
    ):
        if os.path.exists(path):
            return path
    return None


class FaceMatcher:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mtcnn = None
        self.embeddings = None
        self.embeddings_norm = None
        self.names = []
        self.identity_by_image = {}
        self.is_ready = False
        self.error = None
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def load_data(self, on_progress=None):
        def progress(msg):
            if on_progress:
                on_progress(msg)

        try:
            progress("Загрузка FaceNet...")
            self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

            progress("Инициализация детектора лиц...")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=40,
                min_face_size=60,
                thresholds=[0.6, 0.7, 0.8],
                keep_all=False,
                post_process=False,
                device=self.device,
            )

            embeddings_path = "all_embeddings.npy"
            names_path = "image_names.txt"

            if not os.path.exists(embeddings_path):
                raise FileNotFoundError("Не найден файл all_embeddings.npy")
            if not os.path.exists(names_path):
                raise FileNotFoundError("Не найден файл image_names.txt")

            progress("Чтение базы векторов (~400 МБ)...")
            self.embeddings = np.load(embeddings_path).astype(np.float32)
            with open(names_path, "r", encoding="utf-8") as f:
                self.names = [line.strip() for line in f if line.strip()]

            if len(self.names) != len(self.embeddings):
                raise ValueError(
                    f"База повреждена: {len(self.names)} имён, "
                    f"{len(self.embeddings)} векторов"
                )

            identity_path = find_celeba_file("identity_CelebA.txt")
            if identity_path:
                progress("Загрузка идентификаторов личностей...")
                with open(identity_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            self.identity_by_image[parts[0]] = parts[1]

            attr_path = find_celeba_file("list_attr_celeba.txt")
            if attr_path:
                progress("Фильтр: только женские лица...")
                with open(attr_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > 2:
                    headers = lines[1].strip().split()
                    if "Male" in headers:
                        male_idx = headers.index("Male")
                        female_names = set()
                        for line in lines[2:]:
                            parts = line.strip().split()
                            if len(parts) > male_idx + 1 and parts[male_idx + 1] == "-1":
                                female_names.add(parts[0])

                        mask = np.array([name in female_names for name in self.names])
                        if mask.any():
                            self.embeddings = self.embeddings[mask]
                            self.names = [
                                name for name, keep in zip(self.names, mask) if keep
                            ]
            else:
                progress("Атрибуты CelebA не найдены — поиск по всей базе")

            progress("Подготовка индекса...")
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings_norm = self.embeddings / norms

            self.is_ready = True
            self.error = None
            progress(f"Готово · {len(self.names):,} лиц")
        except Exception as exc:
            self.is_ready = False
            self.error = str(exc)
            progress(f"Ошибка: {exc}")
            raise

    def _crop_face_like_celeba(self, image, box):
        x1, y1, x2, y2 = box.astype(int)
        width, height = image.size
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * 0.35)
        margin_y = int(face_h * 0.45)
        left = max(0, x1 - margin_x)
        top = max(0, y1 - margin_y)
        right = min(width, x2 + margin_x)
        bottom = min(height, y2 + margin_y)
        return image.crop((left, top, right, bottom))

    def _extract_embedding(self, cv2_frame):
        image = Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        boxes, probs = self.mtcnn.detect(image)

        if boxes is not None and len(boxes) > 0 and probs is not None and probs[0] >= 0.9:
            face_pil = self._crop_face_like_celeba(image, boxes[0])
        else:
            width, height = image.size
            if width < 80 or height < 80:
                return None
            face_pil = image

        tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def _distance_to_percent(self, distance):
        if distance <= DISTANCE_VERY_CLOSE:
            val = 90 + (DISTANCE_VERY_CLOSE - distance) / DISTANCE_VERY_CLOSE * 10
        elif distance <= DISTANCE_SAME_PERSON:
            ratio = (DISTANCE_SAME_PERSON - distance) / (
                DISTANCE_SAME_PERSON - DISTANCE_VERY_CLOSE
            )
            val = 60 + ratio * 30
        else:
            val = max(0, (1 - distance / 2) * 60)
        return round(float(val), 1)

    def get_top_matches(self, frame, top_n=TOP_N):
        if not self.is_ready or self.embeddings_norm is None:
            return []

        emb = self._extract_embedding(frame)
        if emb is None:
            return []

        distances = np.linalg.norm(self.embeddings_norm - emb, axis=1)
        candidate_indices = np.argsort(distances)[:CANDIDATE_POOL]

        results = []
        seen_identities = set()
        for idx in candidate_indices:
            name = self.names[idx]
            identity = self.identity_by_image.get(name, name)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)

            distance = float(distances[idx])
            results.append((
                str(name),
                self._distance_to_percent(distance),
                distance,
            ))
            if len(results) >= top_n:
                break

        return results


def encode_frame(frame, width=560, height=420):
    small_frame = cv2.resize(frame, (width, height))
    ok, buffer = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def decode_uploaded_image(file_path=None, file_bytes=None):
    if file_bytes:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
    elif file_path:
        arr = np.fromfile(file_path, dtype=np.uint8)
    else:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def celeb_image_path(photo_name):
    for p in (
        os.path.join(CELEBA_IMAGE_DIR, photo_name),
        os.path.join("img_align_celeba", photo_name),
        os.path.join("images", photo_name),
        photo_name,
    ):
        if os.path.exists(p):
            return p
    return None


def load_image_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def main(page: ft.Page):
    page.title = "CelebTwin — Найди своего двойника"
    page.bgcolor = COLORS["bg"]
    page.padding = 0
    page.window.width = 1280
    page.window.height = 860
    page.theme_mode = ft.ThemeMode.DARK

    matcher = FaceMatcher()
    state = {"captured_frame": None, "searching": False, "loading": True}

    CAM_W, CAM_H = 560, 420

    status_text = ft.Text("Загрузка нейросети...", size=12, color=COLORS["muted"])
    status_chip = ft.Container(
        content=ft.Row(
            [
                ft.ProgressRing(width=14, height=14, stroke_width=2, color=COLORS["cyan"]),
                status_text,
            ],
            spacing=8,
        ),
        bgcolor=COLORS["surface_light"],
        border_radius=20,
        padding=ft.Padding.symmetric(vertical=6, horizontal=12),
    )

    photo_view = ft.Image(
        src="",
        width=CAM_W,
        height=CAM_H,
        fit=ft.BoxFit.COVER,
        visible=False,
        gapless_playback=True,
    )

    photo_placeholder = ft.Container(
        width=CAM_W,
        height=CAM_H,
        bgcolor=COLORS["surface_light"],
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.UPLOAD_FILE, size=48, color=COLORS["border"]),
                ft.Text("Фото не загружено", size=15, color=COLORS["text"], weight=ft.FontWeight.W_500),
                ft.Text("Нажмите «Загрузить фото»", size=12, color=COLORS["muted"]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
    )

    search_overlay = ft.Container(
        visible=False,
        width=CAM_W,
        height=CAM_H,
        bgcolor="#CC080C18",
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            [
                ft.ProgressRing(width=48, height=48, stroke_width=3, color=COLORS["purple"]),
                ft.Text("Анализируем лицо...", size=14, color=COLORS["text"], weight=ft.FontWeight.W_500),
                ft.Text("Ищем совпадения в базе", size=11, color=COLORS["muted"]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        ),
    )

    photo_panel = ft.Stack(
        [photo_placeholder, photo_view, search_overlay],
        width=CAM_W,
        height=CAM_H,
    )

    results_column = ft.Column(spacing=12, scroll=ft.ScrollMode.ALWAYS, expand=True)

    def empty_state():
        return ft.Container(
            padding=40,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL, size=56, color=COLORS["border"]),
                    ft.Text(
                        "Ваши двойники появятся здесь",
                        size=16,
                        color=COLORS["text"],
                        weight=ft.FontWeight.W_600,
                    ),
                    ft.Text(
                        "Загрузите фото и нажмите «Найти двойника»",
                        size=13,
                        color=COLORS["muted"],
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
        )

    async def set_status(text, loading=False, success=False, error=False):
        if loading:
            icon = ft.ProgressRing(width=14, height=14, stroke_width=2, color=COLORS["cyan"])
            color = COLORS["muted"]
        elif error:
            icon = ft.Icon(ft.Icons.ERROR_OUTLINE, size=16, color=COLORS["danger"])
            color = COLORS["danger"]
        elif success:
            icon = ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=COLORS["green"])
            color = COLORS["green"]
        else:
            icon = ft.Icon(ft.Icons.INFO_OUTLINE, size=16, color=COLORS["cyan"])
            color = COLORS["muted"]

        status_chip.content = ft.Row([icon, ft.Text(text, size=12, color=color)], spacing=8)
        upload_button.disabled = state["loading"] or not matcher.is_ready
        find_button.disabled = (
            state["loading"]
            or not matcher.is_ready
            or state["captured_frame"] is None
            or state["searching"]
        )
        page.update()

    results_column.controls.append(empty_state())

    def match_bar(percent, color):
        return ft.Container(
            width=160,
            height=6,
            bgcolor=COLORS["border"],
            border_radius=3,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            content=ft.Container(
                width=float(160 * percent / 100),
                height=6,
                bgcolor=color,
                border_radius=3,
            ),
        )

    def rank_style(rank):
        styles = {
            1: (COLORS["gold"], "🥇", True),
            2: (COLORS["silver"], "🥈", False),
            3: (COLORS["bronze"], "🥉", False),
        }
        return styles.get(rank, (COLORS["muted"], f"#{rank}", False))

    def result_card(photo_name, similarity, rank):
        accent, medal, is_hero = rank_style(rank)
        img_path = celeb_image_path(photo_name)
        celeb_b64 = load_image_b64(img_path) if img_path else None
        img_size = 110 if is_hero else 72

        if celeb_b64:
            img_control = ft.Container(
                content=ft.Image(
                    src=celeb_b64,
                    width=img_size,
                    height=img_size,
                    fit=ft.BoxFit.COVER,
                ),
                width=img_size,
                height=img_size,
                border_radius=14 if is_hero else 10,
                border=ft.Border.all(2, accent),
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
            )
        else:
            img_control = ft.Container(
                width=img_size,
                height=img_size,
                bgcolor=COLORS["surface_light"],
                border_radius=14 if is_hero else 10,
                alignment=ft.Alignment.CENTER,
                content=ft.Icon(ft.Icons.PERSON, color=COLORS["muted"], size=32),
            )

        return ft.Container(
            bgcolor=COLORS["surface_light"] if is_hero else COLORS["surface"],
            border_radius=16,
            padding=16 if is_hero else 14,
            border=ft.Border.all(2 if is_hero else 1, accent if is_hero else COLORS["border"]),
            content=ft.Row(
                [
                    img_control,
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(medal, size=18),
                                    ft.Text(
                                        "Лучшее совпадение" if is_hero else f"Совпадение #{rank}",
                                        size=13 if is_hero else 12,
                                        color=accent,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                ],
                                spacing=6,
                            ),
                            ft.Text(
                                photo_name.replace(".jpg", "").replace("_", " "),
                                size=18 if is_hero else 15,
                                color=COLORS["text"],
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Row(
                                [
                                    ft.Text(
                                        f"{similarity:.1f}%",
                                        size=22 if is_hero else 18,
                                        color=accent,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    match_bar(float(similarity), accent),
                                ],
                                spacing=12,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        spacing=6,
                        expand=True,
                    ),
                ],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    def apply_uploaded_frame(frame):
        state["captured_frame"] = frame.copy()
        b64 = encode_frame(frame, CAM_W, CAM_H)
        if b64:
            photo_view.src = b64
            photo_view.visible = True
            photo_placeholder.visible = False
        find_button.disabled = not matcher.is_ready
        page.update()

    async def on_upload_click(e):
        if state["loading"]:
            await set_status("Подождите, нейросеть ещё загружается...", loading=True)
            return
        if not matcher.is_ready:
            await set_status(
                matcher.error or "Нейросеть не загружена. Перезапустите приложение.",
                error=True,
            )
            return

        try:
            files = await file_picker.pick_files(
                dialog_title="Выберите фото",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["jpg", "jpeg", "png"],
                allow_multiple=False,
            )
        except Exception as exc:
            await set_status(f"Не удалось открыть диалог: {exc}", error=True)
            return

        if not files:
            return

        picked = files[0]
        frame = None
        if picked.path:
            frame = decode_uploaded_image(file_path=picked.path)
        elif picked.bytes:
            frame = decode_uploaded_image(file_bytes=picked.bytes)

        if frame is None:
            await set_status("Не удалось прочитать изображение", error=True)
            return

        apply_uploaded_frame(frame)

    async def apply_search_results(matches):
        results_column.controls.clear()
        if matches:
            for i, (photo_name, similarity, _) in enumerate(matches, 1):
                results_column.controls.append(result_card(photo_name, similarity, i))
        else:
            results_column.controls.append(
                ft.Container(
                    bgcolor=COLORS["surface"],
                    border_radius=14,
                    padding=20,
                    border=ft.Border.all(1, COLORS["border"]),
                    content=ft.Column(
                        [
                            ft.Icon(ft.Icons.FACE_6, size=40, color=COLORS["danger"]),
                            ft.Text(
                                "Лицо не обнаружено",
                                size=15,
                                color=COLORS["text"],
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Text(
                                "Загрузите другое фото: лицо по центру, хорошее освещение",
                                size=12,
                                color=COLORS["muted"],
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                )
            )

        state["searching"] = False
        search_overlay.visible = False
        upload_button.disabled = not matcher.is_ready
        find_button.disabled = state["captured_frame"] is None or not matcher.is_ready
        page.update()

    async def on_find_click(e):
        if state["loading"]:
            await set_status("Подождите, нейросеть ещё загружается...", loading=True)
            return
        if not matcher.is_ready:
            await set_status(
                matcher.error or "Нейросеть не загружена",
                error=True,
            )
            return
        if state["captured_frame"] is None:
            await set_status("Сначала загрузите фото", error=True)
            return
        if state["searching"]:
            return

        state["searching"] = True
        search_overlay.visible = True
        find_button.disabled = True
        upload_button.disabled = True
        page.update()

        frame_copy = state["captured_frame"].copy()

        def search():
            try:
                matches = matcher.get_top_matches(frame_copy, top_n=TOP_N)
            except Exception as exc:
                matches = None
                matcher.error = str(exc)

            async def finish():
                if matches is None:
                    await set_status(f"Ошибка поиска: {matcher.error}", error=True)
                    state["searching"] = False
                    search_overlay.visible = False
                    upload_button.disabled = not matcher.is_ready
                    find_button.disabled = False
                    page.update()
                    return
                await apply_search_results(matches)

            page.run_task(finish)

        threading.Thread(target=search, daemon=True).start()

    upload_button = ft.Button(
        "Загрузить фото",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=on_upload_click,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor=COLORS["cyan"],
            color=COLORS["bg"],
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(vertical=12, horizontal=18),
        ),
    )

    find_button = ft.Button(
        "Найти двойника",
        icon=ft.Icons.AUTO_AWESOME,
        on_click=on_find_click,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor=COLORS["purple"],
            color=COLORS["text"],
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(vertical=12, horizontal=18),
        ),
    )

    def instruction_step(num, color, text):
        return ft.Row(
            [
                ft.Container(
                    width=24,
                    height=24,
                    border_radius=12,
                    bgcolor=color,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Text(str(num), size=11, color=COLORS["bg"], weight=ft.FontWeight.BOLD),
                ),
                ft.Text(text, size=13, color=COLORS["muted"]),
            ],
            spacing=10,
        )

    instructions_block = ft.Container(
        bgcolor=COLORS["surface_light"],
        border_radius=14,
        padding=16,
        content=ft.Column(
            [
                ft.Text("Как пользоваться", size=13, color=COLORS["text"], weight=ft.FontWeight.BOLD),
                instruction_step(1, COLORS["green"], "Дождитесь зелёной галочки вверху"),
                instruction_step(2, COLORS["cyan"], "Нажмите «Загрузить фото»"),
                instruction_step(3, COLORS["purple"], "Нажмите «Найти двойника»"),
                instruction_step(4, COLORS["gold"], "Используйте фото, где лицо хорошо видно"),
                instruction_step(5, COLORS["silver"], "Лучше работает с портретами анфас"),
            ],
            spacing=10,
        ),
    )

    left_panel = ft.Container(
        expand=1,
        padding=ft.Padding.only(right=12),
        content=ft.Column(
            [
                ft.Text("Фотография", size=18, weight=ft.FontWeight.BOLD, color=COLORS["text"]),
                ft.Container(
                    content=photo_panel,
                    border_radius=16,
                    border=ft.Border.all(2, COLORS["purple"]),
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                ),
                ft.Row([upload_button, find_button], spacing=10, wrap=True),
                instructions_block,
            ],
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            scroll=ft.ScrollMode.ALWAYS,
            expand=True,
        ),
    )

    right_panel = ft.Container(
        expand=1,
        padding=ft.Padding.only(left=12),
        content=ft.Column(
            [
                ft.Text("Ваши двойники", size=18, weight=ft.FontWeight.BOLD, color=COLORS["text"]),
                ft.Container(
                    expand=True,
                    bgcolor=COLORS["surface"],
                    border_radius=16,
                    padding=16,
                    border=ft.Border.all(1, COLORS["border"]),
                    content=results_column,
                ),
            ],
            spacing=14,
            expand=True,
        ),
    )

    header = ft.Container(
        bgcolor=COLORS["surface"],
        padding=ft.Padding.symmetric(vertical=20, horizontal=28),
        border=ft.Border.only(bottom=ft.BorderSide(1, COLORS["border"])),
        content=ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(
                            width=42,
                            height=42,
                            border_radius=12,
                            bgcolor=COLORS["purple"],
                            alignment=ft.Alignment.CENTER,
                            content=ft.Icon(ft.Icons.FACE_RETOUCHING_NATURAL, color=COLORS["text"], size=22),
                        ),
                        ft.Column(
                            [
                                ft.Text("CelebTwin", size=22, weight=ft.FontWeight.BOLD, color=COLORS["text"]),
                                ft.Text(
                                    "AI-поиск похожих знаменитостей · CelebA",
                                    size=12,
                                    color=COLORS["muted"],
                                ),
                            ],
                            spacing=0,
                        ),
                    ],
                    spacing=14,
                ),
                ft.Container(expand=True),
                status_chip,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    )

    page.add(
        ft.Column(
            [
                header,
                ft.Container(
                    expand=True,
                    padding=ft.Padding.symmetric(vertical=16, horizontal=24),
                    content=ft.Row(
                        [left_panel, ft.VerticalDivider(width=1, color=COLORS["border"]), right_panel],
                        expand=True,
                        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                ),
            ],
            expand=True,
            spacing=0,
        )
    )

    def load_model_worker():
        def on_progress(msg):
            async def update():
                await set_status(msg, loading=True)

            page.run_task(update)

        try:
            matcher.load_data(on_progress=on_progress)

            async def ready():
                state["loading"] = False
                await set_status(
                    f"Готово · {len(matcher.names):,} лиц в базе",
                    success=True,
                )

            page.run_task(ready)
        except Exception:
            async def failed():
                state["loading"] = False
                await set_status(
                    matcher.error or "Не удалось загрузить нейросеть",
                    error=True,
                )

            page.run_task(failed)

    threading.Thread(target=load_model_worker, daemon=True).start()


if __name__ == "__main__":
    ft.run(main)
