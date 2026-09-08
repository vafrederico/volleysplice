from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "screenshots" / "phone-release-sources-v2"
OUTPUT_DIR = ROOT / "screenshots" / "phone-upload-featured"
BACKGROUND = ROOT / "featured-background-v2.png"
LOGO = ROOT.parent / "app" / "src" / "main" / "res" / "drawable-xxxhdpi" / "volleysplice_logo.png"

WIDTH = 1080
HEIGHT = 2160
ORANGE = "#ff5a36"
WHITE = "#ffffff"
MUTED = "#c8d5ea"
NAVY = "#07142d"

BOLD_FONT = Path("C:/Windows/Fonts/arialbd.ttf")
REGULAR_FONT = Path("C:/Windows/Fonts/arial.ttf")

SCREENS = [
    {
        "source": "01-choose-game-window.png",
        "output": "01-choose-game-window.png",
        "step": "1 · SET UP",
        "headline": ("Choose the part", "with the game"),
        "support": "Use the full video, or mark exactly where the game starts and ends.",
        "crop_y": 80,
    },
    {
        "source": "02-fine-tune-settings.png",
        "output": "02-fine-tune-when-needed.png",
        "step": "2 · SETTINGS",
        "headline": ("Fine-tune only", "when you need to"),
        "support": "Automatic cleanup, clip padding, short breaks, and review sensitivity live in Settings.",
        "crop_y": 300,
    },
    {
        "source": "03-save-final-video.png",
        "output": "03-save-finished-video.png",
        "step": "3 · EXPORT",
        "headline": ("Save the", "finished video"),
        "support": "Save the final video when the review looks right—or create YouTube chapters.",
        "crop_y": 800,
    },
    {
        "source": "04-check-score-markers.png",
        "output": "04-check-score-markers.png",
        "step": "4 · SCORE",
        "headline": ("Check the", "score markers"),
        "support": "Correct which side serves, add missed serves, and record team side switches.",
        "crop_y": 550,
    },
    {
        "source": "05-review-attention-queues.png",
        "output": "05-start-with-attention.png",
        "step": "5 · REVIEW",
        "headline": ("Start with what", "needs attention"),
        "support": "Review automatic cleanup, clips, and serves; ignored footage stays out of the queues.",
        "crop_y": 300,
    },
    {
        "source": "06-review-game-timeline.png",
        "output": "06-review-suggested-clips.png",
        "step": "6 · TIMELINE",
        "headline": ("Review the", "suggested clips"),
        "support": "Each timeline block is a clip planned for the final video—select one to check it.",
        "crop_y": 450,
    },
    {
        "source": "07-fix-selected-rally.png",
        "output": "07-fix-one-clip.png",
        "step": "7 · EDIT",
        "headline": ("Fix one", "clip"),
        "support": "Keep or remove a rally, then adjust padding, rally length, or split at the playhead.",
        "crop_y": 350,
    },
    {
        "source": "08-add-missed-rally.png",
        "output": "08-add-missed-rally.png",
        "step": "8 · ADD",
        "headline": ("Add anything", "VolleySplice missed"),
        "support": "Mark a missed rally, or leave out camera gaps, breaks, and other unusable footage.",
        "crop_y": 350,
    },
]


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > 20:
        font = ImageFont.truetype(str(REGULAR_FONT), size)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(str(REGULAR_FONT), size)


def rounded_source(source: Image.Image, crop_y: int, size: tuple[int, int]) -> Image.Image:
    crop_height = round(source.width * size[1] / size[0])
    crop_y = min(max(0, crop_y), max(0, source.height - crop_height))
    crop = source.crop((0, crop_y, source.width, crop_y + crop_height))
    crop = crop.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=34, fill=255)
    crop.putalpha(mask)
    return crop


def compose(item: dict[str, object], index: int) -> Image.Image:
    background = cover(Image.open(BACKGROUND).convert("RGB"), (WIDTH, HEIGHT)).convert("RGBA")
    shade = Image.new("RGBA", (WIDTH, HEIGHT), (3, 12, 31, 74))
    background.alpha_composite(shade)

    logo = Image.open(LOGO).convert("RGB").resize((382, 129), Image.Resampling.LANCZOS)
    logo_mask = Image.new("L", logo.size, 0)
    ImageDraw.Draw(logo_mask).rounded_rectangle((0, 0, logo.width - 1, logo.height - 1), radius=18, fill=255)
    logo_rgba = logo.convert("RGBA")
    logo_rgba.putalpha(logo_mask)
    background.alpha_composite(logo_rgba, (64, 52))

    card_x, card_y = 60, 590
    card_size = (960, 1510)
    shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (card_x + 8, card_y + 18, card_x + card_size[0] + 8, card_y + card_size[1] + 18),
        radius=42,
        fill=(0, 0, 0, 180),
    )
    background.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(20)))

    source = Image.open(SOURCE_DIR / str(item["source"])).convert("RGB")
    screenshot = rounded_source(source, int(item["crop_y"]), card_size)
    background.alpha_composite(screenshot, (card_x, card_y))

    # Create the drawing context only after all alpha composites. Keeping an
    # ImageDraw handle alive across in-place composites can drop queued header
    # pixels on some Pillow builds.
    draw = ImageDraw.Draw(background)
    draw.rounded_rectangle(
        (card_x, card_y, card_x + card_size[0], card_y + card_size[1]),
        radius=42,
        outline=(255, 255, 255, 225),
        width=8,
    )

    count_font = ImageFont.truetype(str(BOLD_FONT), 28)
    count = f"{index} / {len(SCREENS)}"
    count_box = draw.textbbox((0, 0), count, font=count_font)
    count_width = count_box[2] - count_box[0]
    draw.rounded_rectangle((WIDTH - count_width - 108, 82, WIDTH - 56, 139), radius=24, fill=(6, 20, 48, 215))
    draw.text((WIDTH - count_width - 82, 93), count, font=count_font, fill=WHITE)

    step_font = ImageFont.truetype(str(BOLD_FONT), 28)
    draw.text((64, 234), str(item["step"]), font=step_font, fill=ORANGE)

    headline_font = ImageFont.truetype(str(BOLD_FONT), 76)
    line_one, line_two = item["headline"]
    draw.text((64, 278), line_one, font=headline_font, fill=WHITE, stroke_width=1, stroke_fill=NAVY)
    draw.text((64, 357), line_two, font=headline_font, fill=ORANGE, stroke_width=1, stroke_fill=NAVY)

    support = str(item["support"])
    support_font = fit_text(draw, support, 952, 34)
    draw.text((64, 470), support, font=support_font, fill=MUTED)
    return background.convert("RGB")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(SCREENS, start=1):
        output = compose(item, index)
        output.save(OUTPUT_DIR / str(item["output"]), format="PNG", optimize=True)


if __name__ == "__main__":
    main()
