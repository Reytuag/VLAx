"""
Real-time visualization of agent observations during LIBERO task execution with dual camera views.
Uses matplotlib for lightweight, interactive display of both agentview and eye-in-hand camera views.

Features:
    - Real-time display of both camera views side-by-side
    - Overlay of action vectors, rewards, and task info
    - Display of robot state (joint positions)
    - Attention heatmap visualization across diffusion steps and layers
    - Smooth updates without blocking event loop
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, List, Tuple
from collections import deque
import threading
from queue import Queue


class RealtimeAgentVisualizerDual:
    """
    Real-time visualization of agent observations with dual camera views.
    
    Uses matplotlib with non-blocking updates for live monitoring
    of LIBERO environment observations during task execution.
    Displays both agentview and eye-in-hand camera views with attention heatmaps.
    """
    
    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 128,
        show_action_history: bool = True,
        max_history: int = 10,
        update_interval: int = 100,  # milliseconds
        show_attention: bool = False,
        num_layers: int = 3,  # Number of transformer layers to visualize
        cache_layers: Optional[List[str]] = None,  # Layer names for display
        show_dataset_images: bool = False  # Whether to show dataset images for comparison
    ):
        """
        Initialize the visualizer.
        
        Args:
            image_height: Height of observation images
            image_width: Width of observation images
            show_action_history: Whether to show recent action values
            max_history: Maximum number of past actions to display
            update_interval: Update interval in milliseconds
            show_attention: Whether to display attention heatmaps
            num_layers: Number of transformer layers for attention visualization
            cache_layers: List of layer names (e.g., ["layer_9", "layer_26", "layer_31"])
            show_dataset_images: Whether to show dataset images alongside environment images
        """
        self.image_height = image_height
        self.image_width = image_width
        self.show_action_history = show_action_history
        self.max_history = max_history
        self.update_interval = update_interval
        self.show_attention = show_attention
        self.num_layers = num_layers
        self.cache_layers = cache_layers if cache_layers else [f"layer_{i}" for i in range(num_layers)]
        self.show_dataset_images = show_dataset_images
        
        # Data queue for thread-safe updates
        self.data_queue = Queue(maxsize=1)
        
        # Store recent history
        self.action_history = deque(maxlen=max_history)
        self.reward_history = deque(maxlen=max_history)
        self.step_count = 0
        
        # VLM prompt and response
        self.vlm_prompt = None
        self.vlm_response = None
        
        # Figure and axes
        self.fig = None
        self.ax_agentview = None
        self.ax_eye_in_hand = None
        self.ax_dataset_agentview = None  # New: dataset agentview
        self.ax_dataset_eye_in_hand = None  # New: dataset eye-in-hand
        self.ax_info = None
        self.ax_state = None
        self.ax_vlm = None  # New axis for VLM prompt/response
        self.im_agentview = None
        self.im_eye_in_hand = None
        self.im_dataset_agentview = None  # New: dataset agentview image
        self.im_dataset_eye_in_hand = None  # New: dataset eye-in-hand image
        self.text_info = None
        self.text_state = None
        self.text_vlm = None  # New text for VLM display
        
        # Attention visualization components
        self.step_mosaic_axes = {}  # Dictionary to store axes for each (step, layer)
        self.step_mosaic_images = {}  # Dictionary to store image objects
        self.diffusion_steps = [0, 4, 9]  # 1st, 5th, 10th diffusion steps
        self.attention_dims_initialized = False  # Track if dimensions have been set
        
        # Thread safety
        self.is_running = False
        self.lock = threading.Lock()
        
        # Create figure
        self._setup_figure()
    
    def _setup_figure(self):
        """Set up matplotlib figure with proper layout for dual camera views and attention."""
        if self.show_attention:
            # Layout with attention:
            # Top row: [Agentview] [Eye-in-hand] [Info]
            # Second row: [State Info] [VLM Prompt/Response]
            # Bottom rows: [Attention heatmaps mosaic for 3 diffusion steps × 3 layers]
            
            self.fig = plt.figure(figsize=(18, 15), dpi=100)
            self.fig.suptitle('LIBERO Agent Visualization - Dual Camera with Attention', fontsize=12, fontweight='bold')
            
            # Create nested grid spec
            # Outer grid: 3 rows (top cameras + middle info + bottom mosaic)
            gs_outer = gridspec.GridSpec(3, 1, figure=self.fig, 
                                         height_ratios=[1.2, 0.6, 1.5],  # Increased top row height slightly
                                         hspace=0.35)
            
            # Top row subgrid: 2 rows x 2 columns for ENV and DATASET images if show_dataset_images, otherwise 1 row x 3 columns
            if self.show_dataset_images:
                # Layout: Row 1: [ENV agentview, ENV eye-in-hand]
                #         Row 2: [DATASET agentview, DATASET eye-in-hand]
                gs_top = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_outer[0], 
                                                           wspace=0.3, hspace=0.3)
            else:
                # Original layout: [agentview, eye-in-hand, info]
                gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], 
                                                           wspace=0.3)
            
            # Middle row: 2 columns for state info and VLM display
            gs_middle = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1],
                                                          wspace=0.3)
            
            # Bottom row subgrid: 3 columns × 3 rows (3 steps × 3 layers) + colorbar
            gs_mosaic = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_outer[2], 
                                                          hspace=0.25, wspace=0.4)
            
            # ========== TOP ROW: CAMERA VIEWS ==========
            blank_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            
            if self.show_dataset_images:
                # ENV images (row 0)
                self.ax_agentview = self.fig.add_subplot(gs_top[0, 0])
                self.ax_agentview.set_title('ENV: Agent View', fontsize=9, fontweight='bold', color='blue')
                self.ax_agentview.axis('off')
                self.im_agentview = self.ax_agentview.imshow(blank_image, origin='upper')
                
                self.ax_eye_in_hand = self.fig.add_subplot(gs_top[0, 1])
                self.ax_eye_in_hand.set_title('ENV: Eye-in-Hand', fontsize=9, fontweight='bold', color='blue')
                self.ax_eye_in_hand.axis('off')
                self.im_eye_in_hand = self.ax_eye_in_hand.imshow(blank_image, origin='upper')
                
                # Info panel (spans both rows)
                self.ax_info = self.fig.add_subplot(gs_top[:, 2])
                self.ax_info.axis('off')
                self.text_info = self.ax_info.text(
                    0.05, 0.95,
                    'Waiting for data...',
                    transform=self.ax_info.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    family='monospace'
                )
                
                # DATASET images (row 1)
                self.ax_dataset_agentview = self.fig.add_subplot(gs_top[1, 0])
                self.ax_dataset_agentview.set_title('DATASET: Agent View', fontsize=9, fontweight='bold', color='green')
                self.ax_dataset_agentview.axis('off')
                self.im_dataset_agentview = self.ax_dataset_agentview.imshow(blank_image, origin='upper')
                
                self.ax_dataset_eye_in_hand = self.fig.add_subplot(gs_top[1, 1])
                self.ax_dataset_eye_in_hand.set_title('DATASET: Eye-in-Hand', fontsize=9, fontweight='bold', color='green')
                self.ax_dataset_eye_in_hand.axis('off')
                self.im_dataset_eye_in_hand = self.ax_dataset_eye_in_hand.imshow(blank_image, origin='upper')
            else:
                # Original layout without dataset images
                self.ax_agentview = self.fig.add_subplot(gs_top[0, 0])
                self.ax_agentview.set_title('Agent View (Third Person)', fontsize=10, fontweight='bold')
                self.ax_agentview.axis('off')
                self.im_agentview = self.ax_agentview.imshow(blank_image, origin='upper')
                
                self.ax_eye_in_hand = self.fig.add_subplot(gs_top[0, 1])
                self.ax_eye_in_hand.set_title('Eye-in-Hand View (Wrist Camera)', fontsize=10, fontweight='bold')
                self.ax_eye_in_hand.axis('off')
                self.im_eye_in_hand = self.ax_eye_in_hand.imshow(blank_image, origin='upper')
                
                # Info panel
                self.ax_info = self.fig.add_subplot(gs_top[0, 2])
                self.ax_info.axis('off')
                self.text_info = self.ax_info.text(
                    0.05, 0.95,
                    'Waiting for data...',
                    transform=self.ax_info.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    family='monospace'
                )
            
            # ========== MIDDLE ROW: STATE INFO AND VLM DISPLAY ==========
            self.ax_state = self.fig.add_subplot(gs_middle[0, 0])
            self.ax_state.axis('off')
            self.text_state = self.ax_state.text(
                0.05, 0.95,
                'Robot State:\nWaiting for data...',
                transform=self.ax_state.transAxes,
                fontsize=7,
                verticalalignment='top',
                family='monospace'
            )
            
            # VLM Prompt and Response display
            self.ax_vlm = self.fig.add_subplot(gs_middle[0, 1])
            self.ax_vlm.axis('off')
            self.text_vlm = self.ax_vlm.text(
                0.05, 0.95,
                'VLM Interaction:\nWaiting for data...',
                transform=self.ax_vlm.transAxes,
                fontsize=7,
                verticalalignment='top',
                family='monospace',
                wrap=True
            )
            
            # ========== BOTTOM ROW: ATTENTION MOSAIC ==========
            # Dynamic layout: 3 columns (diffusion steps) × num_layers rows
            
            step_labels = ['Step 1', 'Step 5', 'Step 10']
            
            # Create blank attention for initialization (will be updated dynamically)
            # Start with a reasonable default size: 17 queries, 256 VLM cache + 17 action = 273 keys
            blank_attn = np.zeros((17, 273))
            
            # Dynamically create grid spec based on num_layers
            gs_mosaic = gridspec.GridSpecFromSubplotSpec(self.num_layers, 4, subplot_spec=gs_outer[2], 
                                                          hspace=0.25, wspace=0.4)
            
            # Create rows for each layer
            for layer_idx in range(self.num_layers):
                layer_name = self.cache_layers[layer_idx] if layer_idx < len(self.cache_layers) else f"Layer {layer_idx}"
                for step_col, (step_idx, step_label) in enumerate(zip(self.diffusion_steps, step_labels)):
                    ax = self.fig.add_subplot(gs_mosaic[layer_idx, step_col])
                    ax.set_title(f'{step_label}-{layer_name}', fontsize=7, fontweight='bold')
                    im = ax.imshow(blank_attn, cmap='hot', aspect='auto', vmin=0, vmax=1)
                    ax.set_xlabel('Key Index', fontsize=5)
                    ax.set_ylabel('Query Index', fontsize=5)
                    ax.tick_params(labelsize=4)
                    
                    # Store for later updates (rectangles will be added when first data arrives)
                    self.step_mosaic_axes[(step_idx, layer_idx)] = ax
                    self.step_mosaic_images[(step_idx, layer_idx)] = im
            
            # Add colorbar in the rightmost column
            ax_cbar = self.fig.add_subplot(gs_mosaic[:, 3])
            ax_cbar.axis('off')
            # Create a mappable object for the colorbar
            sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=ax_cbar, pad=0.1, fraction=0.8, aspect=20)
            cbar.set_label('Attn Weight', fontsize=7, labelpad=10)
            cbar.ax.tick_params(labelsize=6)
            
            plt.tight_layout()
        else:
            # Layout without attention
            if self.show_dataset_images:
                # Layout: 4 rows, 2 columns
                # Row 1: [Env Agentview] [Env Eye-in-hand]
                # Row 2: [Dataset Agentview] [Dataset Eye-in-hand]
                # Row 3: [Info] [State]
                # Row 4: [VLM] [VLM]
                self.fig = plt.figure(figsize=(14, 16), dpi=100)
                self.fig.suptitle('LIBERO Agent Visualization - Env vs Dataset Comparison', fontsize=14, fontweight='bold')
                
                # Environment images (row 1)
                self.ax_agentview = plt.subplot(4, 2, 1)
                self.ax_agentview.set_title('ENV: Agent View', fontsize=11, fontweight='bold', color='blue')
                self.ax_agentview.axis('off')
                blank_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                self.im_agentview = self.ax_agentview.imshow(blank_image, origin='upper')
                
                self.ax_eye_in_hand = plt.subplot(4, 2, 2)
                self.ax_eye_in_hand.set_title('ENV: Eye-in-Hand View', fontsize=11, fontweight='bold', color='blue')
                self.ax_eye_in_hand.axis('off')
                self.im_eye_in_hand = self.ax_eye_in_hand.imshow(blank_image, origin='upper')
                
                # Dataset images (row 2)
                self.ax_dataset_agentview = plt.subplot(4, 2, 3)
                self.ax_dataset_agentview.set_title('DATASET: Agent View', fontsize=11, fontweight='bold', color='green')
                self.ax_dataset_agentview.axis('off')
                self.im_dataset_agentview = self.ax_dataset_agentview.imshow(blank_image, origin='upper')
                
                self.ax_dataset_eye_in_hand = plt.subplot(4, 2, 4)
                self.ax_dataset_eye_in_hand.set_title('DATASET: Eye-in-Hand View', fontsize=11, fontweight='bold', color='green')
                self.ax_dataset_eye_in_hand.axis('off')
                self.im_dataset_eye_in_hand = self.ax_dataset_eye_in_hand.imshow(blank_image, origin='upper')
                
                # Info and state (row 3)
                self.ax_info = plt.subplot(4, 2, 5)
                self.ax_info.axis('off')
                self.text_info = self.ax_info.text(
                    0.05, 0.95,
                    'Waiting for data...',
                    transform=self.ax_info.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace'
                )
                
                self.ax_state = plt.subplot(4, 2, 6)
                self.ax_state.axis('off')
                self.text_state = self.ax_state.text(
                    0.05, 0.95,
                    'Robot State:\nWaiting for data...',
                    transform=self.ax_state.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace'
                )
                
                # VLM (row 4, spans both columns)
                self.ax_vlm = plt.subplot(4, 1, 4)
                self.ax_vlm.axis('off')
                self.text_vlm = self.ax_vlm.text(
                    0.05, 0.95,
                    'VLM Interaction:\nWaiting for data...',
                    transform=self.ax_vlm.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    family='monospace',
                    wrap=True
                )
            else:
                # Original layout without dataset images (3 rows)
                self.fig = plt.figure(figsize=(14, 12), dpi=100)
                self.fig.suptitle('LIBERO Agent Visualization - Dual Camera View', fontsize=14, fontweight='bold')
                
                # Create grid: 3 rows, 2 columns
                self.ax_agentview = plt.subplot(3, 2, 1)
                self.ax_agentview.set_title('Agent View (Third Person)', fontsize=11, fontweight='bold')
                self.ax_agentview.axis('off')
                blank_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                self.im_agentview = self.ax_agentview.imshow(blank_image, origin='upper')
                
                self.ax_eye_in_hand = plt.subplot(3, 2, 2)
                self.ax_eye_in_hand.set_title('Eye-in-Hand View (Wrist Camera)', fontsize=11, fontweight='bold')
                self.ax_eye_in_hand.axis('off')
                self.im_eye_in_hand = self.ax_eye_in_hand.imshow(blank_image, origin='upper')
                
                self.ax_info = plt.subplot(3, 2, 3)
                self.ax_info.axis('off')
                self.text_info = self.ax_info.text(
                    0.05, 0.95,
                    'Waiting for data...',
                    transform=self.ax_info.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace'
                )
                
                self.ax_state = plt.subplot(3, 2, 4)
                self.ax_state.axis('off')
                self.text_state = self.ax_state.text(
                    0.05, 0.95,
                    'Robot State:\nWaiting for data...',
                    transform=self.ax_state.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    family='monospace'
                )
                
                # VLM Prompt and Response display (spans both columns in row 3)
                self.ax_vlm = plt.subplot(3, 1, 3)
                self.ax_vlm.axis('off')
                self.text_vlm = self.ax_vlm.text(
                    0.05, 0.95,
                    'VLM Interaction:\nWaiting for data...',
                    transform=self.ax_vlm.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    family='monospace',
                    wrap=True
                )
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def show(self, block: bool = False):
        """
        Show the visualization window.
        
        Args:
            block: Whether to block execution until window is closed
        """
        self.is_running = True
        plt.ion()  # Enable interactive mode
        plt.show(block=block)
        if not block:
            plt.pause(0.001)  # Small pause to ensure window appears
    
    def _initialize_attention_rectangles(self, query_len: int, vlm_cache_length: int, action_self_attn_length: int):
        """
        Initialize highlighting rectangles for all attention visualizations (called once).
        
        Highlights the separation between the first 256 VLM cache keys and the last 17 action keys.
        
        Args:
            query_len: Number of queries
            vlm_cache_length: Length of VLM cache (expected 256)
            action_self_attn_length: Length of action self-attention (expected 17)
        """
        from matplotlib.patches import Rectangle
        
        total_key_length = vlm_cache_length + action_self_attn_length
        
        # Initialize rectangles for all (step, layer) combinations
        for (step_idx, layer_idx), ax in self.step_mosaic_axes.items():
            # Add green dotted contour around the full attention map
            rect_full = Rectangle((-0.5, -0.5), total_key_length, query_len,
                                  linewidth=2, edgecolor='lime', facecolor='none', linestyle=':', alpha=0.8)
            ax.add_patch(rect_full)
            
            # Update axis labels with actual dimensions
            ax.set_xlabel(f'Key (0-{total_key_length})', fontsize=5)
            ax.set_ylabel(f'Query (0-{query_len})', fontsize=5)
        
        self.attention_dims_initialized = True
    
    def _process_attention_weights(self, attn_weights) -> Optional[Tuple[list, dict]]:
        """
        Process attention weights from flow model into visualization.
        
        Converts attention weights to (query_len, total_key_length) for visualization.
        Automatically infers dimensions from the attention map shape.
        Now handles multi-layer attention weights.
        
        Args:
            attn_weights: Attention weights (potentially from JAX array)
                         Can have shape:
                         - (batch, num_layers, heads, query_len, key_len) - multiple layers
                         - (batch, heads, query_len, key_len) - single layer
            
        Returns:
            Tuple of (list of 2D numpy arrays, dict with dimension info) or None
            - List: one array per layer, each (query_len, total_key_len) suitable for heatmap visualization
            - Dict: {'query_len': int, 'total_key_length': int, 'vlm_cache_length': int, 'action_self_attn_length': int}
        """
        try:
            # Convert JAX array to numpy if needed
            if hasattr(attn_weights, '__array__'):
                attn_weights = np.array(attn_weights)
            
            # Convert JAX array to numpy if needed
            if hasattr(attn_weights, '__array__'):
                attn_weights = np.array(attn_weights)
            
            # Determine if we have multiple layers
            shape = attn_weights.shape
            num_layers = 1
            layer_weights = [attn_weights]
            
            # Check if this has a layer dimension (batch, num_layers, ...)
            if len(shape) == 5:
                # Shape: (batch, num_layers, heads, query_len, key_len)
                num_layers = shape[1]
                layer_weights = [attn_weights[:, i, :, :, :] for i in range(num_layers)]
            elif len(shape) == 4:
                # Could be (batch, heads, query_len, key_len) - single layer
                # Or (batch, num_layers, query_len, key_len) - depends on context
                # Try to infer: if shape[1] is small (< 10), likely num_heads, else likely num_layers
                if shape[1] <= 10:  # Likely heads
                    layer_weights = [attn_weights]
                else:  # Likely layers
                    num_layers = shape[1]
                    layer_weights = [attn_weights[:, i, :, :] for i in range(num_layers)]
            
            processed_layers = []
            
            for layer_idx, attn in enumerate(layer_weights):
                # Process each layer's attention weights
                if len(attn.shape) == 4:
                    # Shape: (batch, heads, query_len, key_len)
                    # Average over all heads, take first batch
                    attn = attn[0].mean(axis=0)  # (query_len, key_len)
                elif len(attn.shape) == 3:
                    # Shape: (batch, query_len, key_len) or (heads, query_len, key_len)
                    attn = attn[0]  # Take first batch or head
                elif len(attn.shape) == 2:
                    # Already (query_len, key_len)
                    pass
                else:
                    # Try flattening and reshaping - infer dimensions from original shape
                    # Use the last two dimensions as query_len and key_len
                    original_shape = attn.shape
                    if len(original_shape) >= 2:
                        query_len_infer = original_shape[-2]
                        key_len_infer = original_shape[-1]
                        attn = attn.reshape((query_len_infer, key_len_infer))
                    else:
                        continue  # Skip this layer if we can't reshape
                
                # Now infer dimensions from the processed attention map
                query_len = attn.shape[0]
                total_key_length = attn.shape[1]
                
                # Fixed boundary: first 256 keys are VLM cache, last 17 are action self-attention
                vlm_cache_length = 256
                action_self_attn_length = total_key_length - vlm_cache_length
                
                # Ensure dimensions are reasonable
                if action_self_attn_length < 0:
                    # If total key length is less than 256, assume no fixed boundary
                    vlm_cache_length = 0
                    action_self_attn_length = total_key_length
                
                # Clip negative values to 0 (attention weights are non-negative)
                attn = np.clip(attn, 0, None)
                
                # Use min-max normalization to utilize full colormap range
                attn_min = attn.min()
                attn_max = attn.max()
                
                if attn_max > attn_min:
                    # Standard min-max normalization: maps [min, max] -> [0, 1]
                    attn = (attn - attn_min) / (attn_max - attn_min)
                elif attn_max > 0:
                    # If all non-zero values are the same, still normalize
                    attn = attn / attn_max
                else:
                    # All zeros - no change needed
                    pass
                
                processed_layers.append(attn.astype(np.float32))
            
            # Return both processed layers and dimension info from the first layer
            if processed_layers:
                # Get dimensions from first processed layer
                first_layer = processed_layers[0]
                query_len = first_layer.shape[0]
                total_key_length = first_layer.shape[1]
                
                # Fixed boundary: first 256 keys are VLM cache, last 17 are action self-attention
                vlm_cache_length = 256
                action_self_attn_length = total_key_length - vlm_cache_length
                
                if action_self_attn_length < 0:
                    vlm_cache_length = 0
                    action_self_attn_length = total_key_length
                
                dim_info = {
                    'query_len': query_len,
                    'total_key_length': total_key_length,
                    'vlm_cache_length': vlm_cache_length,
                    'action_self_attn_length': action_self_attn_length
                }
                
                return (processed_layers, dim_info)
            else:
                return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def update(
        self,
        agentview_image: Optional[np.ndarray] = None,
        eye_in_hand_image: Optional[np.ndarray] = None,
        dataset_agentview_image: Optional[np.ndarray] = None,  # New
        dataset_eye_in_hand_image: Optional[np.ndarray] = None,  # New
        robot_state: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        task_info: Optional[str] = None,
        step: Optional[int] = None,
        diffusion_step_attentions: Optional[dict] = None,
        vlm_prompt: Optional[str] = None,
        vlm_response: Optional[str] = None,
        **kwargs
    ):
        """
        Update the visualization with new data.
        
        Args:
            agentview_image: RGB image from environment agentview camera (H, W, 3)
            eye_in_hand_image: RGB image from environment eye-in-hand camera (H, W, 3)
            dataset_agentview_image: RGB image from dataset agentview (H, W, 3)
            dataset_eye_in_hand_image: RGB image from dataset eye-in-hand (H, W, 3)
            robot_state: Robot joint positions (9D vector)
            action: Action taken (7D vector)
            reward: Reward received
            done: Whether episode is done
            task_info: Task description text
            step: Current step number
            diffusion_step_attentions: Dict mapping diffusion step index to attention weights (optional)
            vlm_prompt: VLM prompt text (optional)
            vlm_response: VLM response text (optional)
        """
        with self.lock:
            # Update VLM prompt and response if provided
            if vlm_prompt is not None:
                self.vlm_prompt = vlm_prompt
            if vlm_response is not None:
                self.vlm_response = vlm_response
            
            # Update environment images
            if agentview_image is not None:
                self.im_agentview.set_data(agentview_image)
            
            if eye_in_hand_image is not None:
                self.im_eye_in_hand.set_data(eye_in_hand_image)
            
            # Update dataset images (if showing)
            if self.show_dataset_images:
                if dataset_agentview_image is not None and self.im_dataset_agentview is not None:
                    self.im_dataset_agentview.set_data(dataset_agentview_image)
                
                if dataset_eye_in_hand_image is not None and self.im_dataset_eye_in_hand is not None:
                    self.im_dataset_eye_in_hand.set_data(dataset_eye_in_hand_image)
            
            # Update diffusion step mosaic if available
            if self.show_attention and diffusion_step_attentions is not None and isinstance(diffusion_step_attentions, dict):
                for step_idx, step_attn in diffusion_step_attentions.items():
                    result = self._process_attention_weights(step_attn)
                    if result is not None:
                        attn_viz_layers, dim_info = result
                        
                        # Initialize rectangles on first update
                        if not self.attention_dims_initialized:
                            self._initialize_attention_rectangles(
                                dim_info['query_len'],
                                dim_info['vlm_cache_length'],
                                dim_info['action_self_attn_length']
                            )
                        
                        # Update the attention heatmaps
                        for layer_idx in range(len(attn_viz_layers)):
                            key = (step_idx, layer_idx)
                            if key in self.step_mosaic_images:
                                self.step_mosaic_images[key].set_data(attn_viz_layers[layer_idx])
            
            # Update step count
            if step is not None:
                self.step_count = step
            else:
                self.step_count += 1
            
            # Update action and reward history
            if action is not None:
                self.action_history.append(action)
            if reward is not None:
                self.reward_history.append(reward)
            
            # Build info text
            info_lines = []
            
            if task_info:
                info_lines.append(f"Task: {task_info}")
                info_lines.append("")
            
            info_lines.append(f"Step: {self.step_count}")
            
            if reward is not None:
                total_reward = sum(self.reward_history)
                info_lines.append(f"Reward (current): {reward:.4f}")
                info_lines.append(f"Reward (total): {total_reward:.4f}")
            
            if done is not None:
                info_lines.append(f"Done: {done}")
            
            info_lines.append("")
            
            # Show recent actions
            if self.show_action_history and len(self.action_history) > 0:
                info_lines.append("Recent Actions (last 5):")
                for i, act in enumerate(list(self.action_history)[-5:]):
                    action_str = f"  {i+1}: [{', '.join([f'{a:.3f}' for a in act[:3]])}..."
                    info_lines.append(action_str)
            
            self.text_info.set_text('\n'.join(info_lines))
            
            # Build robot state text
            state_lines = ["Robot State (Joint Positions):"]
            state_lines.append("")
            
            if robot_state is not None and len(robot_state) > 0:
                # Display joint positions in a readable format
                for i in range(0, len(robot_state), 3):
                    joint_values = robot_state[i:i+3]
                    joint_str = f"  Joints {i}-{i+len(joint_values)-1}: "
                    joint_str += ", ".join([f"{v:.4f}" for v in joint_values])
                    state_lines.append(joint_str)
            else:
                state_lines.append("  No state data available")
            
            self.text_state.set_text('\n'.join(state_lines))
            
            # Build VLM interaction text
            vlm_lines = ["VLM Interaction:"]
            vlm_lines.append("=" * 60)
            
            if self.vlm_prompt is not None:
                vlm_lines.append("")
                vlm_lines.append("PROMPT:")
                # Wrap prompt text
                prompt_wrapped = self._wrap_text(self.vlm_prompt, width=80)
                vlm_lines.extend(prompt_wrapped)
            else:
                vlm_lines.append("")
                vlm_lines.append("PROMPT: No VLM prompt yet")
            
            if self.vlm_response is not None:
                vlm_lines.append("")
                vlm_lines.append("RESPONSE:")
                # Wrap response text
                response_wrapped = self._wrap_text(self.vlm_response, width=80)
                vlm_lines.extend(response_wrapped)
            else:
                vlm_lines.append("")
                vlm_lines.append("RESPONSE: No VLM response yet")
            
            if hasattr(self, 'text_vlm') and self.text_vlm is not None:
                self.text_vlm.set_text('\n'.join(vlm_lines))
    
    def _wrap_text(self, text: str, width: int = 80) -> list:
        """
        Wrap text to specified width.
        
        Args:
            text: Text to wrap
            width: Maximum line width
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = "  "  # Indent
        
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line == "  ":
                    current_line += word
                else:
                    current_line += " " + word
            else:
                lines.append(current_line)
                current_line = "  " + word
        
        if current_line.strip():
            lines.append(current_line)
        
        return lines if lines else ["  (empty)"]
    
    def refresh(self, pause_time: float = 0.001):
        """
        Force a refresh of the display.
        
        Args:
            pause_time: Time to pause for display update (seconds)
        """
        if self.is_running:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(pause_time)
    
    def render_to_array(self) -> np.ndarray:
        """
        Render the current figure to a numpy RGB array.
        
        Returns:
            RGB image as numpy array (H, W, 3) with uint8 dtype
        """
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        # Convert RGBA to RGB
        return img[:, :, :3].copy()
    
    def close(self):
        """Close the visualization window."""
        self.is_running = False
        plt.close(self.fig)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class SimpleImageViewer:
    """
    Simple static image viewer for displaying single frames.
    Useful for debugging and quick visualization.
    """
    
    def __init__(self, title: str = "Image Viewer"):
        """
        Initialize the viewer.
        
        Args:
            title: Window title
        """
        self.title = title
        self.fig = None
        self.ax = None
        self.im = None
    
    def show(self, image: np.ndarray, block: bool = True):
        """
        Display an image.
        
        Args:
            image: RGB image (H, W, 3)
            block: Whether to block execution
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.suptitle(self.title, fontsize=12, fontweight='bold')
            self.ax.axis('off')
            self.im = self.ax.imshow(image)
        else:
            self.im.set_data(image)
        
        plt.draw()
        plt.pause(0.001)
        
        if block:
            plt.show()
    
    def close(self):
        """Close the viewer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None


def render_attention_video_frame(
    agentview_image: np.ndarray,
    eye_in_hand_image: np.ndarray,
    task_description: str,
    step: int,
    reward: float = 0.0,
    robot_state: Optional[np.ndarray] = None,
    attention_weights: Optional[dict] = None,
    cache_layers: Optional[List[str]] = None,
    vlm_cache_length: int = 256,
    fig_ref: Optional[dict] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Render a single video frame with dual camera views and attention heatmaps.
    
    Creates a matplotlib figure matching the realtime visualization layout,
    renders it to a numpy array, and returns it. The figure is reused across
    calls for performance via the fig_ref dict.
    
    Args:
        agentview_image: RGB image from agentview camera (H, W, 3)
        eye_in_hand_image: RGB image from eye-in-hand camera (H, W, 3)
        task_description: Task description text
        step: Current step number
        reward: Current reward
        robot_state: Robot state vector (9D)
        attention_weights: Dict mapping diffusion step index to attention weights
        cache_layers: List of layer names (e.g., ["layer_9", "layer_26", "layer_31"])
        vlm_cache_length: Number of VLM cache keys (boundary for highlighting)
        fig_ref: Mutable dict to persist figure state between calls.
                 Pass {} on first call; it will be populated with reusable objects.
    
    Returns:
        Tuple of (frame_array, fig_ref):
        - frame_array: RGB image as numpy array (H, W, 3) with uint8 dtype
        - fig_ref: Updated fig_ref dict for reuse
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for rendering
    
    if cache_layers is None:
        cache_layers = ["layer_9", "layer_26", "layer_31"]
    num_layers = len(cache_layers)
    diffusion_steps_to_show = [0, 4, 9]
    step_labels = ['Step 1', 'Step 5', 'Step 10']
    
    # ---- First-time setup: create figure and axes ----
    if fig_ref is None:
        fig_ref = {}
    
    if 'fig' not in fig_ref:
        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.suptitle('Trajectory Visualization with Attention Maps', fontsize=12, fontweight='bold')
        
        gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.5], hspace=0.30)
        
        # Top row: [agentview] [eye-in-hand] [info]
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], wspace=0.3)
        
        ax_agent = fig.add_subplot(gs_top[0, 0])
        ax_agent.set_title('Agent View', fontsize=10, fontweight='bold')
        ax_agent.axis('off')
        im_agent = ax_agent.imshow(agentview_image, origin='upper')
        
        ax_eye = fig.add_subplot(gs_top[0, 1])
        ax_eye.set_title('Eye-in-Hand View', fontsize=10, fontweight='bold')
        ax_eye.axis('off')
        im_eye = ax_eye.imshow(eye_in_hand_image, origin='upper')
        
        ax_info = fig.add_subplot(gs_top[0, 2])
        ax_info.axis('off')
        text_info = ax_info.text(
            0.05, 0.95, '', transform=ax_info.transAxes,
            fontsize=8, verticalalignment='top', family='monospace'
        )
        
        # Bottom: attention mosaic  (num_layers rows × 3 columns + colorbar column)
        gs_mosaic = gridspec.GridSpecFromSubplotSpec(
            num_layers, 4, subplot_spec=gs_outer[1], hspace=0.30, wspace=0.40
        )
        
        attn_axes = {}
        attn_images = {}
        blank_attn = np.zeros((17, 273))  # Default placeholder
        
        for layer_idx in range(num_layers):
            layer_name = cache_layers[layer_idx]
            for step_col, (step_idx, step_label) in enumerate(zip(diffusion_steps_to_show, step_labels)):
                ax = fig.add_subplot(gs_mosaic[layer_idx, step_col])
                ax.set_title(f'{step_label}-{layer_name}', fontsize=7, fontweight='bold')
                im = ax.imshow(blank_attn, cmap='hot', aspect='auto', vmin=0, vmax=1)
                ax.set_xlabel('Key Index', fontsize=5)
                ax.set_ylabel('Query Index', fontsize=5)
                ax.tick_params(labelsize=4)
                attn_axes[(step_idx, layer_idx)] = ax
                attn_images[(step_idx, layer_idx)] = im
        
        # Colorbar
        ax_cbar = fig.add_subplot(gs_mosaic[:, 3])
        ax_cbar.axis('off')
        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_cbar, pad=0.1, fraction=0.8, aspect=20)
        cbar.set_label('Attn Weight', fontsize=7, labelpad=10)
        cbar.ax.tick_params(labelsize=6)
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        fig_ref['fig'] = fig
        fig_ref['im_agent'] = im_agent
        fig_ref['im_eye'] = im_eye
        fig_ref['text_info'] = text_info
        fig_ref['attn_axes'] = attn_axes
        fig_ref['attn_images'] = attn_images
        fig_ref['rects_initialized'] = False
    
    # ---- Update data ----
    fig = fig_ref['fig']
    fig_ref['im_agent'].set_data(agentview_image)
    fig_ref['im_eye'].set_data(eye_in_hand_image)
    
    # Info text
    info_lines = [
        f"Task: {task_description}",
        "",
        f"Step: {step}",
        f"Reward: {reward:.4f}",
    ]
    if robot_state is not None:
        info_lines.append("")
        info_lines.append("Robot State:")
        for i in range(0, len(robot_state), 3):
            vals = robot_state[i:i+3]
            info_lines.append(f"  [{i}-{i+len(vals)-1}]: " + ", ".join(f"{v:.3f}" for v in vals))
    fig_ref['text_info'].set_text('\n'.join(info_lines))
    
    # ---- Update attention heatmaps ----
    if attention_weights is not None and isinstance(attention_weights, dict):
        # Use the same processing logic as the realtime visualizer
        for step_idx, step_attn in attention_weights.items():
            # Convert JAX array to numpy if needed
            attn_np = np.array(step_attn) if hasattr(step_attn, '__array__') else step_attn
            
            shape = attn_np.shape
            # Handle (batch, num_layers, heads, q, k) or (batch, heads, q, k)
            if len(shape) == 5:
                layer_list = [attn_np[:, i, :, :, :] for i in range(shape[1])]
            elif len(shape) == 4 and shape[1] <= 10:
                layer_list = [attn_np]
            elif len(shape) == 4:
                layer_list = [attn_np[:, i, :, :] for i in range(shape[1])]
            else:
                layer_list = [attn_np]
            
            for layer_idx, attn in enumerate(layer_list):
                # Reduce to 2D
                if len(attn.shape) == 4:
                    attn = attn[0].mean(axis=0)
                elif len(attn.shape) == 3:
                    attn = attn[0]
                
                attn = np.clip(attn, 0, None)
                a_min, a_max = attn.min(), attn.max()
                if a_max > a_min:
                    attn = (attn - a_min) / (a_max - a_min)
                elif a_max > 0:
                    attn = attn / a_max
                
                key = (step_idx, layer_idx)
                if key in fig_ref['attn_images']:
                    fig_ref['attn_images'][key].set_data(attn.astype(np.float32))
        
        # Draw separator rectangles once after first real data
        if not fig_ref['rects_initialized'] and attention_weights:
            from matplotlib.patches import Rectangle
            first_key = next(iter(attention_weights))
            first_attn = np.array(attention_weights[first_key]) if hasattr(attention_weights[first_key], '__array__') else attention_weights[first_key]
            # Infer query_len from the attention shape
            if len(first_attn.shape) == 5:
                query_len = first_attn.shape[-2]
                total_key_len = first_attn.shape[-1]
            elif len(first_attn.shape) == 4:
                query_len = first_attn.shape[-2]
                total_key_len = first_attn.shape[-1]
            else:
                query_len = first_attn.shape[-2]
                total_key_len = first_attn.shape[-1]
            
            action_self_attn_length = total_key_len - vlm_cache_length
            
            for (si, li), ax in fig_ref['attn_axes'].items():
                # Add green dotted contour around the full attention map
                rect_full = Rectangle((-0.5, -0.5), total_key_len, query_len,
                                      linewidth=2, edgecolor='lime', facecolor='none', linestyle=':', alpha=0.8)
                ax.add_patch(rect_full)
                ax.set_xlabel(f'Key (0-{total_key_len})', fontsize=5)
                ax.set_ylabel(f'Query (0-{query_len})', fontsize=5)
            fig_ref['rects_initialized'] = True
    
    # ---- Render to array ----
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3].copy()
    
    return frame, fig_ref
