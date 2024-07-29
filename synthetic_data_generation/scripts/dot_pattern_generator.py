from PIL import Image, ImageDraw
import random

# TODO: ADD THRESHOLDING FOR DISTANCE

def create_randomized_grid_dot_pattern(image_size, base_distance_mm, dot_radius_mm, dpi):
    # Convert mm to pixels
    base_distance_px = int(base_distance_mm / 25.4 * dpi)
    dot_radius_px = int(dot_radius_mm / 25.4 * dpi)
    
    scale_factor = 4
    
    # Create a high-resolution image for anti-aliasing
    high_res_image_size = (image_size[0] * scale_factor, image_size[1] * scale_factor)
    high_res_image = Image.new("RGB", high_res_image_size, (0, 0, 0))
    draw = ImageDraw.Draw(high_res_image)
    
    # Adjust dot radius and distance for high-resolution image
    high_res_dot_radius_px = dot_radius_px * scale_factor
    high_res_base_distance_px = base_distance_px * scale_factor
    
    # Generate a single random distance to be used consistently across the entire image
    distance = high_res_base_distance_px + random.randint(-high_res_base_distance_px // 2, high_res_base_distance_px // 2)
    col_distance = high_res_base_distance_px + random.randint(-high_res_base_distance_px // 2, high_res_base_distance_px // 2)
    
    # Draw the dots in a grid pattern with consistent random distances
    current_y = high_res_dot_radius_px
    while current_y < high_res_image_size[1] - high_res_dot_radius_px:
        current_x = high_res_dot_radius_px
        while current_x < high_res_image_size[0] - high_res_dot_radius_px:
            draw.ellipse(
                (current_x - high_res_dot_radius_px, current_y - high_res_dot_radius_px, 
                 current_x + high_res_dot_radius_px, current_y + high_res_dot_radius_px), 
                fill=(255, 255, 255)
            )
            current_x += distance
        current_y += distance
    
    # Downsample the high-resolution image to the desired resolution
    image = high_res_image.resize(image_size, Image.ANTIALIAS)
    
    return image

#Example Usage
if __name__ == "__main__":
    image_size = (1024, 1024)  # Image size in pixels
    base_distance_mm = 5       # Base distance in mm
    dot_radius_mm = random.uniform(.5, 1)          # Dot radius in mm
    dpi = 300                  # Dots per inch for conversion

    dot_pattern = create_randomized_grid_dot_pattern(image_size, base_distance_mm, dot_radius_mm, dpi)
    dot_pattern.save("../images/randomized_grid_dot_pattern_inverted.png")
