# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from glob import glob
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
import numpy as np
import pandas as pd

class DataAnything3DataApp:
    """
    Standalone Gradio app for visualizing Depth-Anything-3 results from the output directory.
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        
    def scan_output_folders(self) -> List[str]:
        """
        Scan the output directory for subfolders that contain results.
        A folder is considered a result folder if it contains 'metrics.json', 'depth_vis/', or 'scene.glb'.
        """
        result_folders = []
        if not os.path.exists(self.output_dir):
            return []
            
        for root, dirs, files in os.walk(self.output_dir):
            has_results = (
                "metrics.json" in files or 
                "scene.glb" in files or
                os.path.isdir(os.path.join(root, "depth_vis")) or
                os.path.isdir(os.path.join(root, "gs_ply"))
            )
            if has_results:
                rel_path = os.path.relpath(root, self.output_dir)
                if rel_path == ".":
                    continue
                result_folders.append(rel_path)
        
        return sorted(result_folders, key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)) if os.path.exists(os.path.join(self.output_dir, x)) else 0, reverse=True)

    def load_folder_data(self, folder_rel_path: str) -> Tuple[pd.DataFrame, List[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Load data from the selected folder.
        """
        if not folder_rel_path:
            return pd.DataFrame(), [], None, None, None, None
            
        full_path = os.path.join(self.output_dir, folder_rel_path)
        
        # Load Metrics
        metrics_df = pd.DataFrame()
        metrics_path = os.path.join(full_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
                # Filter out metadata for the table
                table_data = {k: v for k, v in metrics_data.items() if isinstance(v, (int, float, str))}
                metrics_df = pd.DataFrame([table_data])

        # Load Depth Visualizations
        depth_images = sorted(glob(os.path.join(full_path, "depth_vis", "*.jpg")))
        
        # Load Trajectory Plot
        traj_plot = os.path.join(full_path, "trajectory_plot.png")
        if not os.path.exists(traj_plot):
            traj_plot = None
            
        # Load GS Video
        gs_videos = sorted(glob(os.path.join(full_path, "gs_video", "*.mp4")))
        gs_video = gs_videos[0] if gs_videos else None
        
        # Load GS PLY
        gs_plies = sorted(glob(os.path.join(full_path, "gs_ply", "*.ply")))
        gs_ply = gs_plies[0] if gs_plies else None
        
        # Load Scene GLB (Preferred for viewer)
        scene_glb = os.path.join(full_path, "scene.glb")
        if not os.path.exists(scene_glb):
            scene_glb = None
            
        return metrics_df, depth_images, traj_plot, gs_video, gs_ply, scene_glb

    def create_app(self):
        """Create and configure the Gradio app."""
        with gr.Blocks(title="Depth Anything 3 Data Viewer") as app:
            gr.Markdown("# 🔍 Depth Anything 3 Data Viewer")
            gr.Markdown("Browse and visualize reconstruction results from the `output/` directory.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    folder_selector = gr.Dropdown(
                        choices=self.scan_output_folders(),
                        label="Select Result Folder",
                        info="Folders containing results in the output/ directory",
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄 Refresh Folders")
                
                with gr.Column(scale=3):
                    metrics_table = gr.DataFrame(
                        label="Evaluation Metrics",
                        interactive=False
                    )

            with gr.Tabs():
                with gr.Tab("🖼️ Depth Sequence"):
                    depth_gallery = gr.Gallery(
                        label="Depth Visualizations",
                        columns=5,
                        height="auto",
                        preview=True,
                        object_fit="contain"
                    )
                
                with gr.Tab("📈 Trajectory"):
                    traj_image = gr.Image(label="Trajectory Plot", interactive=False)
                
                with gr.Tab("✨ Gaussian Splatting"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gs_video_comp = gr.Video(label="Rendered Video", interactive=False)
                        with gr.Column(scale=3):
                            # Prioritize scene.glb for 3D viewer, fallback to gs_ply if possible
                            load_3d_btn = gr.Button("🚀 Load 3D Viewer", variant="primary")
                            gs_3d_info = gr.Markdown("")
                            gs_3d_viewer = gr.Model3D(label="3D Viewer", height=500)
                            
                            with gr.Row():
                                gs_download_ply = gr.File(label="Download .ply (3DGS)")
                                gs_download_glb = gr.File(label="Download .glb (Scene)")

            # Internal state to store the paths without loading them into components immediately
            current_viewer_file = gr.State(None)

            def update_folder_list():
                folders = self.scan_output_folders()
                return gr.update(choices=folders)

            def handle_selection(folder):
                metrics, images, traj, video, ply, glb = self.load_folder_data(folder)
                viewer_file = glb if glb else ply # Try GLB first for better Gradio support
                
                info_msg = ""
                if viewer_file:
                    size_mb = os.path.getsize(viewer_file) / (1024 * 1024)
                    info_msg = f"📦 **3D File Ready**: `{os.path.basename(viewer_file)}` ({size_mb:.1f} MB)"
                    if size_mb > 50:
                        info_msg += "\n⚠️ *Large file detected. Loading might take a moment and freeze the browser briefly.*"
                else:
                    info_msg = "❌ No 3D model found in this folder."

                return [
                    metrics,
                    images,
                    traj,
                    video,
                    gr.update(value=None), # Reset viewer on folder change
                    info_msg,
                    ply,
                    glb,
                    viewer_file # Update state
                ]

            def load_3d_viewer(viewer_file):
                if not viewer_file:
                    return gr.update(value=None)
                return gr.update(value=viewer_file)

            refresh_btn.click(fn=update_folder_list, outputs=folder_selector)
            
            folder_selector.change(
                fn=handle_selection,
                inputs=folder_selector,
                outputs=[
                    metrics_table,
                    depth_gallery,
                    traj_image,
                    gs_video_comp,
                    gs_3d_viewer,
                    gs_3d_info,
                    gs_download_ply,
                    gs_download_glb,
                    current_viewer_file
                ]
            )

            load_3d_btn.click(
                fn=load_3d_viewer,
                inputs=current_viewer_file,
                outputs=gs_3d_viewer
            )

        return app

def launch_data_gui(share=False):
    app = DataAnything3DataApp().create_app()
    app.launch(share=share)

if __name__ == "__main__":
    launch_data_gui()
