import numpy as np
from typing import Tuple
from PIL import Image

def process_uploaded_image(pil_image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Process an uploaded PIL image for use in genetic algorithm.
    
    Args:
        pil_image: PIL Image object
        target_size: Target dimensions (height, width)
        
    Returns:
        np.ndarray: Processed grayscale image
    """
    # Convert to grayscale if not already
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Resize to target size
    height, width = target_size
    pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(pil_image, dtype=np.float32)
    
    return image_array


def generate_pattern_image(pattern_type: str, size: Tuple[int, int]) -> np.ndarray:
    """
    Generate various pattern images for testing.
    
    Args:
        pattern_type: Type of pattern to generate
        size: Image dimensions (height, width)
        
    Returns:
        np.ndarray: Generated pattern image
    """
    height, width = size
    
    if pattern_type == 'circles':
        return _create_circles_pattern(height, width)
    elif pattern_type == 'squares':
        return _create_squares_pattern(height, width)
    elif pattern_type == 'gradient':
        return _create_gradient_pattern(height, width)
    elif pattern_type == 'noise':
        return _create_noise_pattern(height, width)
    elif pattern_type == 'checkerboard':
        return _create_checkerboard_pattern(height, width)
    else:
        # Default to circles
        return _create_circles_pattern(height, width)


def _create_circles_pattern(height: int, width: int) -> np.ndarray:
    """Create a pattern with concentric circles."""
    image = np.zeros((height, width))
    center_y, center_x = height // 2, width // 2
    
    # Create multiple circles
    y, x = np.ogrid[:height, :width]
    
    # Outer circle
    outer_radius = min(height, width) // 3
    outer_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= outer_radius ** 2
    image[outer_circle] = 150
    
    # Middle circle
    middle_radius = min(height, width) // 5
    middle_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= middle_radius ** 2
    image[middle_circle] = 200
    
    # Inner circle
    inner_radius = min(height, width) // 8
    inner_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= inner_radius ** 2
    image[inner_circle] = 255
    
    return image.astype(np.float32)


def _create_squares_pattern(height: int, width: int) -> np.ndarray:
    """Create a pattern with nested squares."""
    image = np.zeros((height, width))
    center_y, center_x = height // 2, width // 2
    
    # Outer square
    outer_size = min(height, width) // 2
    y1 = max(0, center_y - outer_size // 2)
    y2 = min(height, center_y + outer_size // 2)
    x1 = max(0, center_x - outer_size // 2)
    x2 = min(width, center_x + outer_size // 2)
    image[y1:y2, x1:x2] = 100
    
    # Middle square
    middle_size = min(height, width) // 3
    y1 = max(0, center_y - middle_size // 2)
    y2 = min(height, center_y + middle_size // 2)
    x1 = max(0, center_x - middle_size // 2)
    x2 = min(width, center_x + middle_size // 2)
    image[y1:y2, x1:x2] = 200
    
    # Inner square
    inner_size = min(height, width) // 6
    y1 = max(0, center_y - inner_size // 2)
    y2 = min(height, center_y + inner_size // 2)
    x1 = max(0, center_x - inner_size // 2)
    x2 = min(width, center_x + inner_size // 2)
    image[y1:y2, x1:x2] = 255
    
    return image.astype(np.float32)


def _create_gradient_pattern(height: int, width: int) -> np.ndarray:
    """Create a gradient pattern."""
    # Create linear gradients
    x_gradient = np.linspace(0, 255, width)
    y_gradient = np.linspace(0, 255, height)
    
    # Create 2D gradient
    X, Y = np.meshgrid(x_gradient, y_gradient)
    
    image = X
    
    return image.astype(np.float32)


def _create_noise_pattern(height: int, width: int) -> np.ndarray:
    """Create a random noise pattern."""
    # Generate different types of noise
    noise_type = np.random.choice(['uniform', 'gaussian', 'salt_pepper'])
    
    if noise_type == 'uniform':
        image = np.random.uniform(0, 255, (height, width))
    elif noise_type == 'gaussian':
        image = np.random.normal(127, 64, (height, width))
        image = np.clip(image, 0, 255)
    else:  # salt and pepper
        image = np.random.choice([0, 255], size=(height, width))
    
    return image.astype(np.float32)


def _create_checkerboard_pattern(height: int, width: int) -> np.ndarray:
    """Create a checkerboard pattern."""
    image = np.zeros((height, width))
    
    # Determine checkerboard size (adaptive to image size)
    check_size = max(1, min(height, width) // 8)
    
    for i in range(0, height, check_size):
        for j in range(0, width, check_size):
            # Alternate between black and white
            if ((i // check_size) + (j // check_size)) % 2 == 0:
                end_i = min(i + check_size, height)
                end_j = min(j + check_size, width)
                image[i:end_i, j:end_j] = 255
    
    return image.astype(np.float32)


def validate_image_data(image: np.ndarray) -> bool:
    """
    Validate that image data is properly formatted.
    
    Args:
        image: Image array to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if image.ndim != 2:
        return False
    
    if image.dtype not in [np.float32, np.float64, np.uint8]:
        return False
    
    if np.any(image < 0) or np.any(image > 255):
        return False
    
    return True


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    
    Args:
        image: Input image
        
    Returns:
        np.ndarray: Normalized image
    """
    if image.max() <= 1.0:
        # Assume it's in 0-1 range
        return (image * 255).astype(np.float32)
    else:
        # Assume it's already in 0-255 range
        return np.clip(image, 0, 255).astype(np.float32)


def add_noise_to_image(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image
        noise_level: Standard deviation of noise as fraction of image range
        
    Returns:
        np.ndarray: Noisy image
    """
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.float32)


def create_image_thumbnail(image: np.ndarray, max_size: int = 64) -> np.ndarray:
    """
    Create a thumbnail version of an image.
    
    Args:
        image: Input image
        max_size: Maximum dimension for thumbnail
        
    Returns:
        np.ndarray: Thumbnail image
    """
    height, width = image.shape
    
    # Calculate scaling factor
    scale = min(max_size / height, max_size / width)
    
    if scale >= 1.0:
        return image  # No need to downscale
    
    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Simple nearest neighbor downsampling
    step_y = height / new_height
    step_x = width / new_width
    
    thumbnail = np.zeros((new_height, new_width))
    
    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i * step_y)
            orig_j = int(j * step_x)
            thumbnail[i, j] = image[orig_i, orig_j]
    
    return thumbnail.astype(np.float32)


def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate basic statistics of an image.
    
    Args:
        image: Input image
        
    Returns:
        dict: Image statistics
    """
    return {
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'median': float(np.median(image)),
        'shape': image.shape,
        'size': image.size
    }