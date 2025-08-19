import numpy as np
import cv2
import random
from PIL import Image, ImageEnhance

class FFTSuppressAmplitude:
    """
    Suppress the amplitude component of image, modulated by ampl_scale argument. (Lower ampl_scale -> Higher amplitude removal)
    """
    def __init__(self, ampl_scale=0):
        """
        Args:
            ampl_scale (float): Scale factor for amplitude modification.
                                Typical values: 0 - 5, 7, 10
        """
        self.ampl_scale = ampl_scale

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Input RGB image.
        Returns:
            PIL.Image: Image reconstructed from FFT with modified amplitude.
        """
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)

        reconstructed_channels = []
        
        for c in range(3):  # RGB channels
            channel = img_np[:, :, c]

            # FFT 
            fft = np.fft.fft2(channel)
            fft_shifted = np.fft.fftshift(fft)
            
            # Amplitude & Phase
            ampl = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            # Replace amplitude with constant
            const_amp = np.ones_like(ampl) + ampl * 1e-5 * self.ampl_scale
            
            # Reconstruct with constant amplitude & original phase
            modified_fft = const_amp * np.exp(1j * phase)
            
            # Inverse FFT
            ifft_shifted = np.fft.ifftshift(modified_fft)
            reconstructed = np.fft.ifft2(ifft_shifted)
            reconstructed = np.abs(reconstructed)

            # Normalize to 0–255
            reconstructed = 255 * (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
            reconstructed_channels.append(reconstructed.astype(np.uint8))

        # Merge channels back
        reconstructed_img = np.stack(reconstructed_channels, axis=2)

        return Image.fromarray(reconstructed_img.astype(np.uint8))
    
class GaussianBlur:
    """
    Apply Gaussian blur with adjustable blur strength
    """
    def __init__(self, blur_scale=10):
        """
        Args:
            blur_scale (float): Scale factor for blur strength modification.
                                Typical values: 5, 10, 25, 50
        """
        self.blur_scale = blur_scale
        
    def __call__(self, img):
        """
        Args:
            img (PIL.Image or np.ndarray): Input RGB image.
        Returns:
            PIL.Image: Image reconstructed from FFT with modified amplitude.
        """
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
    
        if self.blur_scale <= 0:
            return img_np  # No blur

        # Kernel size should be odd and depend on blur scale
        ksize = int(max(3, (self.blur_scale // 2) * 2 + 1))  # Ensure odd kernel
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=self.blur_scale)
        
        return Image.fromarray(blurred.astype(np.uint8))
    
class CannyEdge:
    """
    Edges of images and various sensitivitiy levels
    """
    def __init__(self, edge_scale):
        """
        Args:
            edge_scale (float): Scale factor for edge detection. More sensitive for lower value
                                Typical values: 100 - 500 (increments of 50)
        """
        self.edge_scale = edge_scale
        
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
        
        # Convert to grayscale for Canny
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Define thresholds based on edge_scale
        lower_thresh = max(0, self.edge_scale * 0.5)
        upper_thresh = self.edge_scale

        # Apply Canny edge detector
        edges = cv2.Canny(gray, threshold1=lower_thresh, threshold2=upper_thresh)

        # Convert single-channel edges to 3-channel RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb.astype(np.uint8))
    
class EdgeMaskedBlur:
    """
    Blurring done after masking edges
    """    
    def __init__(self, blur_scale=250, edge_scale=250):
        """
        Args:
            blur_scale (float): Defaults to 250. Typical values: 5, 10, 25, 50
            edge_scale (float): _description_. Defaults to 250. Typical values: 100 - 500 (increments of 50)
        """
        self.blur_scale = blur_scale
        self.edge_scale = edge_scale
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)

        # --- Step 1: Edge detection ---
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lower_thresh = max(0, self.edge_scale * 0.5)
        upper_thresh = self.edge_scale
        edges = cv2.Canny(gray, threshold1=lower_thresh, threshold2=upper_thresh)

        # Create mask: edges = 1, non-edges = 0
        mask = (edges > 0).astype(np.uint8)

        # --- Step 2: Apply Gaussian blur ---
        ksize = int(max(3, (self.blur_scale // 2) * 2 + 1))  # odd kernel size
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=self.blur_scale)

        # --- Step 3: Combine blurred and original using mask ---
        # Expand mask to 3 channels
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # Invert mask so edges = 1 in original image, 0 in blurred
        inv_mask_3c = 1 - mask_3c

        # Keep edges from original image, blur elsewhere
        output = (img_np * mask_3c) + (blurred * inv_mask_3c)
        output = output.astype(np.uint8)
        
        return Image.fromarray(output.astype(np.uint8))
    
class GrayScale:
    """
    Grayscale image
    """
    def __init__(self):
        pass
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
            
        gray_rgb = np.stack([img_np] * 3, axis=2)
        
        return Image.fromarray(gray_rgb.astype(np.uint8))
    
class ColorSwap:
    """
    Random shuffling color channels to get a different color input
    """
    def __init__(self, seed=None):
        self.seed = seed
            
    def __call__(self, img):
        if self.seed is not None:
            random.seed(self.seed)
            
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
                
        channels = [0, 1, 2]
        random.shuffle(channels)
        shuffled_img = img_np[:, :, channels]
        return Image.fromarray(shuffled_img.astype(np.uint8))

class RandomDropChannel:
    """
    Drop one / two selected channels according to a scale
    """
    def __init__(self, drop_rate=1, seed=None):
        self.drop_rate = drop_rate
        self.seed = seed
        
    def __call__(self, img):
        if self.seed is not None:
            random.seed(self.seed)
         
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
            
        num_drop = random.choice([1, 2])  # Drop either 1 or 2 channels
        channels_to_drop = random.sample([0, 1, 2], num_drop)

        for ch in channels_to_drop:
            constant_val = random.randint(0, 255)
            img_np[:, :, ch] = np.clip(constant_val * self.drop_rate + img_np[:, :, ch] * (1 - self.drop_rate), 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
class MyColorJitter:
    """
    Change brightness, hue, contrast, saturation.
    """
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, seed=None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.seed = seed
        
    def __call__(self, img):
        if self.seed is not None:
            random.seed(self.seed)
            
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            img_np = np.array(img, dtype=np.float32)
        else:
            img_np = img.astype(np.float32)
            
        if self.brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-self.contrast, self.contrast))
        if self.saturation > 0:
            img = ImageEnhance.Color(img).enhance(1 + random.uniform(-self.saturation, self.saturation))
        if self.hue > 0:
            hsv = np.array(img.convert("HSV"), dtype=np.uint8)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + int(random.uniform(-self.hue*255, self.hue*255))) % 255

        img_np = np.array(img, dtype=np.float32)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
class HistEqualization:
    """
    Histogram equalization
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        """
        Apply CLAHE to an RGB image.

        Args:
            img (PIL.Image.Image or np.ndarray): Input RGB image.

        Returns:
            PIL.Image.Image: CLAHE-processed RGB image.
        """
        # Ensure numpy array in RGB
        if isinstance(img, Image.Image):
            img = np.array(img.convert('RGB'))
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:  # Grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))

        # Convert back to RGB
        img_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_out.astype(np.uint8))
    
class RandomChannelNormalization:
    def __init__(self, mean_shift_range=(-50, 50), scale_range=(0.5, 1.5), seed=None):
        """
        Randomly normalize each RGB channel by applying scaling and mean shift.

        Args:
            mean_shift_range (tuple): Range for random mean shift per channel.
            scale_range (tuple): Range for random scale factor per channel.
            seed (int, optional): Random seed for reproducibility.
        """
        self.mean_shift_range = mean_shift_range
        self.scale_range = scale_range
        self.seed = seed

    def __call__(self, img):
        """
        Apply random channel normalization.

        Args:
            img (PIL.Image.Image or np.ndarray): Input RGB image.

        Returns:
            PIL.Image.Image: Randomly channel-normalized image.
        """
        if self.seed is not None:
            random.seed(self.seed)

        # Convert to NumPy array
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert('RGB'), dtype=np.float32)
        else:
            img_np = img.astype(np.float32)

        for c in range(3):
            mean_shift = random.uniform(*self.mean_shift_range)
            scale_factor = random.uniform(*self.scale_range)
            img_np[:, :, c] = np.clip(
                (img_np[:, :, c] - img_np[:, :, c].mean()) * scale_factor + mean_shift,
                0, 255
            )

        return Image.fromarray(img_np.astype(np.uint8))
    
class LowFrequencyNoiseInjection:
    def __init__(self, alpha=0.3, blur_kernel=51, seed=None):
        """
        Inject smooth low-frequency noise into an image.

        Args:
            alpha (float): Noise blending strength [0,1].
            blur_kernel (int): Size of Gaussian blur kernel (must be odd).
            seed (int, optional): Random seed for reproducibility.
        """
        self.alpha = alpha
        self.blur_kernel = blur_kernel
        self.seed = seed

    def __call__(self, img):
        """
        Apply low-frequency noise injection.

        Args:
            img (PIL.Image.Image or np.ndarray): Input RGB image.

        Returns:
            PIL.Image.Image: Augmented image with low-frequency noise.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Convert to NumPy array
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert('RGB'), dtype=np.float32)
        else:
            img_np = img.astype(np.float32)

        # Generate random noise in [0,255]
        noise = np.random.uniform(0, 255, img_np.shape).astype(np.float32)

        # Apply Gaussian blur for smoothness
        noise = cv2.GaussianBlur(noise, (self.blur_kernel, self.blur_kernel), 0)

        # Blend with original
        blended = np.clip((1 - self.alpha) * img_np + self.alpha * noise, 0, 255)

        return Image.fromarray(blended.astype(np.uint8))