import numpy as np
from typing import Tuple

def create_default_test_image(size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    """
    Create a default test pattern for image reconstruction.
    
    Args:
        size: Image dimensions (height, width)
        
    Returns:
        np.ndarray: Grayscale test image
    """
    height, width = size
    image = np.zeros((height, width))
    
    # Create a simple geometric pattern
    center_y, center_x = height // 2, width // 2
    
    # Add a circle
    y, x = np.ogrid[:height, :width]
    circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(height, width) // 4) ** 2
    image[circle_mask] = 200
    
    # Add a rectangle
    rect_h, rect_w = height // 4, width // 4
    start_y, start_x = center_y - rect_h // 2, center_x - rect_w // 2
    end_y, end_x = start_y + rect_h, start_x + rect_w
    image[start_y:end_y, start_x:end_x] = 100
    
    # Add some noise for complexity
    noise = np.random.normal(0, 10, (height, width))
    image = np.clip(image + noise, 0, 255)
    
    return image.astype(np.float32)


def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        float: MSE value
    """
    return np.mean((image1 - image2) ** 2)


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_pixel_value: float = 255.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        image1: First image
        image2: Second image
        max_pixel_value: Maximum possible pixel value
        
    Returns:
        float: PSNR in dB
    """
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def normalize_fitness(mse: float, max_mse: float = 255.0**2) -> float:
    """
    Normalize MSE to fitness value in [0, 1].
    
    Args:
        mse: Mean Squared Error
        max_mse: Maximum possible MSE
        
    Returns:
        float: Normalized fitness (higher is better)
    """
    return 1.0 - (mse / max_mse)


