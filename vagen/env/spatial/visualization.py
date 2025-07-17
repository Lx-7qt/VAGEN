import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Union, Optional


def visualize_turns(
    turns: List[Dict],
    image_key: Optional[str] = None,
    spacing: int = 10,
    background_color: tuple = (255, 255, 255)
) -> Union[List[Image.Image], Image.Image]:
    """
    Given a list of turn dictionaries of the form:
        {
            'obs_str': str,
            'prompt': str,            # alternative key
            'multi_modal_data': { placeholder: [PIL.Image, ...], ... }
        }
    Produce per-turn images and a combined mosaic.

    Args:
        turns: List of observation dicts.
        image_key: If provided, only use that key in multi_modal_data for images;
                   otherwise take the first list of PIL.Images found.
        spacing: Pixel spacing between text and images, and between images.
        background_color: RGB tuple for canvas background.
        font_path: Optional path to a .ttf font file. Uses DejaVuSans if None.
        font_size: Font size in points for rendering text.

    Returns:
        Either a list of PIL.Image (one per turn), or a single merged PIL.Image
        stacking all turns vertically.
    """
    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 64)
    except Exception:
        font = ImageFont.load_default()

    per_turn_images: List[Image.Image] = []
    # Iterate turns
    for turn in turns:
        # Wrap non-dict
        if not isinstance(turn, dict):
            turn = {'obs_str': str(turn), 'multi_modal_data': {}}
        # Extract text
        text = turn.get('obs_str') or turn.get('prompt', '')
        # Extract images
        imgs = []
        modal = turn.get('multi_modal_data', {})
        # Single image
        if isinstance(modal, Image.Image):
            imgs = [modal]
        # List of images
        elif isinstance(modal, list) and modal and isinstance(modal[0], Image.Image):
            imgs = modal
        # Dict of lists
        elif isinstance(modal, dict):
            if image_key and image_key in modal and isinstance(modal[image_key], list):
                imgs = modal[image_key]
            else:
                for v in modal.values():
                    if isinstance(v, list) and v and isinstance(v[0], Image.Image):
                        imgs = v
                        break
        if not imgs:
            # blank placeholder
            imgs = [Image.new('RGB', (64, 64), background_color)]

        # Prepare text measurement
        # Split lines
        lines = text.splitlines() or ['']
        dummy = Image.new('RGB', (1, 1))
        measurer = ImageDraw.Draw(dummy)
        text_width = 0
        text_height = 0
        line_sizes = []
        for line in lines:
            bbox = measurer.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            text_width = max(text_width, w)
            text_height += h
            line_sizes.append((w, h))

        # Images layout
        img_widths = [im.width for im in imgs]
        img_heights = [im.height for im in imgs]
        total_img_width = sum(img_widths) + spacing * (len(imgs) - 1)
        max_img_height = max(img_heights)

        # Canvas dimensions
        canvas_width = max(text_width, total_img_width)
        canvas_height = text_height + spacing + max_img_height
        canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(canvas)

        # Draw text lines
        y = 0
        for idx, line in enumerate(lines):
            draw.text((0, y), line, fill=(0, 0, 0), font=font)
            y += line_sizes[idx][1]

        # Paste images
        x = (canvas_width - total_img_width) // 2
        y = text_height + spacing
        for im in imgs:
            canvas.paste(im, (x, y))
            x += im.width + spacing

        per_turn_images.append(canvas)

    # Merge into one mosaic
    if not per_turn_images:
        return []
    total_h = sum(im.height for im in per_turn_images) + spacing * (len(per_turn_images) - 1)
    max_w = max(im.width for im in per_turn_images)
    mosaic = Image.new('RGB', (max_w, total_h), background_color)
    y = 0
    for im in per_turn_images:
        mosaic.paste(im, (0, y))
        y += im.height + spacing

    return mosaic


if __name__ == "__main__":
    # Simple test for visualize_turns utility
    from PIL import Image
    # Create dummy images
    img1 = Image.new('RGB', (80, 60), (255, 0, 0))
    img2 = Image.new('RGB', (80, 60), (0, 255, 0))
    # Build turns list
    turns = [
        {'obs_str': 'First turn observation text', 'multi_modal_data': {'image': [img1]}},
        {'obs_str': 'Second turn observation text', 'multi_modal_data': {'image': [img2, img1]}},
    ]
    # Generate visualization
    mosaic = visualize_turns(turns, image_key='image', font_size=32)
    # Save outputs
    os.makedirs('test_output', exist_ok=True)
    path = os.path.join('test_output', 'trajectory_mosaic.png')
    mosaic.save(path)
    print(f"Saved {path}")
