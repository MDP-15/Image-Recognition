def check_bounding_box(xywh,
                       frame_height,
                       frame_width,
                       height_to_width_ratio=0.6,
                       width_to_height_ratio=0.6,
                       left_right_edge=10,
                       top_bottom_edge=10):
    x, y, width, height = xywh

    height_to_width_ratio = height_to_width_ratio / frame_height * frame_width
    width_to_height_ratio = width_to_height_ratio / frame_width * frame_height

    left_right_edge = left_right_edge / frame_width
    top_bottom_edge = top_bottom_edge / frame_width

    if height / width < height_to_width_ratio or width / height < width_to_height_ratio:
        return False, 'shape'

    half_width = width / 2
    half_height = height / 2

    top_left = (x - half_width, y - half_height)
    bottom_right = (x + half_width, y + half_height)

    if (top_left[0] < left_right_edge or bottom_right[0] > 1 - left_right_edge
       or top_left[1] < top_bottom_edge or bottom_right[1] > 1 - top_bottom_edge):
        return False, 'edge'
    return True, 'ok'
