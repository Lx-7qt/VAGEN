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
            'multi_modal_data': { placeholder: [PIL.Image, ...], ... }
        }
    Produce per-turn images and a combined mosaic.

    Args:
        turns: List of observation dicts.
        image_key: If provided, only use that key in multi_modal_data for images;
                   otherwise take the first list of PIL.Images found.
        spacing: Pixel spacing between text and images, and between images.
        background_color: RGB tuple for canvas background.

    Returns:
        Either a list of PIL.Image (one per turn), or a single merged PIL.Image
        stacking all turns vertically.
    """
    font = ImageFont.truetype("DejaVuSans.ttf", 65)
    per_turn_images: List[Image.Image] = []
    per_turn_images: List[Image.Image] = []

    # Helper to render one turn
    for turn in turns:
        # support non-dict entries by wrapping them
        if not isinstance(turn, dict):
            turn = {'prompt': str(turn), 'multi_modal_data': {}}
        # begin per-turn rendering
        # 1) extract text
        text = turn.get('prompt', '')
        # 2) extract image list
        imgs = []
        modal = turn.get('multi_modal_data', {})
        # Handle different formats of multi_modal_data
        from PIL import Image as _PILImage
        # If modal is a single PIL image, wrap it
        if isinstance(modal, _PILImage.Image):
            imgs = [modal]

        # compute sizes
        # text size
        dummy = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy)
        text_lines = text.splitlines() or ['']
        text_width = 0
        text_height = 0
        line_heights = []
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            text_width = max(text_width, w)
            text_height += h
            line_heights.append(h)

        # images combined width and max height
        img_widths = [im.width for im in imgs]
        img_heights = [im.height for im in imgs]
        total_img_width = sum(img_widths) + spacing * (len(imgs) - 1)
        max_img_height = max(img_heights)

        # canvas size
        canvas_width = max(text_width, total_img_width)
        canvas_height = text_height + spacing + max_img_height

        # create turn canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(canvas)

        # draw text
        y_offset = 0
        for i, line in enumerate(text_lines):
            draw.text((0, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += line_heights[i]

        # paste images centered below text
        x_offset = (canvas_width - total_img_width) // 2
        y_offset = text_height + spacing
        for im in imgs:
            canvas.paste(im, (x_offset, y_offset))
            x_offset += im.width + spacing

        per_turn_images.append(canvas)

    # now merge vertically
    if not per_turn_images:
        return []
    # single tall canvas
    total_height = sum(img.height for img in per_turn_images) + spacing * (len(per_turn_images) - 1)
    max_width = max(img.width for img in per_turn_images)
    merged = Image.new('RGB', (max_width, total_height), background_color)

    y = 0
    for img in per_turn_images:
        merged.paste(img, (0, y))
        y += img.height + spacing

    return merged

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
    merged = visualize_turns(turns, image_key='image')
    # Save outputs
    os.makedirs('test_output', exist_ok=True)
    # Save individual turn images
    per_turn = [merged] if not isinstance(merged, list) else merged
    for idx, im in enumerate(per_turn, 1):
        path = os.path.join('test_output', f'turn_{idx}.png')
        im.save(path)
        print(f"Saved test_output/turn_{idx}.png")
    print("visualize_turns test completed.")
