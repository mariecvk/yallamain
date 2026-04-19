import os
from vedo import *
from pathlib import Path
import math
import time

def create_video(
    output_dir="output",
    file_prefix="file_",
    start_timestep=0,
    end_timestep=None,
    video_name="video_output",
    skip_frames=50,
    fps=1,
    point_size=80,
    show_axes=False,
):
    """
    Generates a video from a series of VTK files, visualizing simulation data.

    Arguments:
        output_dir (str): Directory containing the VTK files. Defaults to "output".
        file_prefix (str): Prefix of the VTK filenames (e.g., "file_" for file_0.vtk).
        start_timestep (int): The first timestep to include in the video. Defaults to 0.
        end_timestep (int, optional): The last timestep to include in the video.
                                     If None, uses the last available timestep.
        video_name (str): Name of the output video file (without extension).
        skip_frames (int): Number of frames to skip between rendered frames to reduce video length.
        fps (int): Frames per second for the output video.
        point_size (int): Size of the points in the visualization.
        show_axes (bool): Whether to display coordinate axes in the video.

    Returns:
        pathlib.Path: The path to the generated video file.

    Error Raises:
        FileNotFoundError: If the output directory or no VTK files are found.
        ValueError: If no timesteps can be extracted from filenames or no files
                    are found within the selected timestep range.
    """
    start_time = time.time()

    output_path = Path(output_dir)
    # Check if the output directory exists
    if not output_path.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {output_path.resolve()}")

    # Get all VTK files matching the prefix
    vtk_files = sorted(output_path.glob(f"{file_prefix}*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(
            f"Keine VTK-Dateien gefunden in {output_path.resolve()} mit Prefix '{file_prefix}'"
        )

    def extract_timestep(path):
        # Helper function to extract the timestep number from the filename
        stem = path.stem
        number = stem.replace(file_prefix, "")
        try:
            return int(number)
        except ValueError:
            return None

    timestep_map = {}
    for f in vtk_files:
        ts = extract_timestep(f)
        if ts is not None:
            timestep_map[ts] = f

    if not timestep_map:
        raise ValueError("Es konnten keine Timesteps aus den Dateinamen gelesen werden.")

    available_timesteps = sorted(timestep_map.keys())

    if end_timestep is None:
        end_timestep = available_timesteps[-1]

    # Filter timesteps based on start, end, and skip_frames
    selected_timesteps = [
        ts for ts in available_timesteps
        if start_timestep <= ts <= end_timestep
    ][::skip_frames]

    if not selected_timesteps:
        raise ValueError("Keine Dateien im gewählten Timestep-Bereich gefunden.")

    video_path = output_path / f"{video_name}.mp4"

    # Initialize Vedo plotter for offscreen rendering
    plt = Plotter(size=(2000, 2000), bg="white", offscreen=True)

    # Set camera properties for consistent view
    plt.camera.SetPosition(0, 0, 30)
    plt.camera.SetFocalPoint(4, 0, 0)
    plt.camera.SetViewUp(0, 1, 0)

    # Define axes for visualization
    axes = Axes(
        xrange=(0, 20),
        yrange=(0, 20),
        xtitle="X",
        ytitle="Y",
        axes_linewidth=2,
        grid_linewidth=2,
        c="k",
        number_of_divisions=10,
    )
    axes.pos(0, 0, -2)

    # Build a lookup table (colormap) for point data
    lut = build_lut(
        [
            (0.0, "#d3ddf7"), # Color for value 0.0
            (1.0, "#117e07"), # Color for value 1.0
            (2.0, "#000000"), # Color for value 2.0
        ],
        vmin=-0.5,
        vmax=3.0,
        below_color="white",
        above_color="black",
        nan_color="red",
        interpolate=False,
    )

    # Initialize video writer
    video = Video(str(video_path), fps=fps, backend="ffmpeg")

    total_frames = len(selected_timesteps)
    print(f"Gefundene Dateien: {len(vtk_files)}")
    print(f"Verwendete Frames: {total_frames}")
    print(f"Video-Ausgabe: {video_path.resolve()}")
    print()

    # Loop through selected timesteps to generate video frames
    for frame_num, timestep in enumerate(selected_timesteps):
        vtk_file = timestep_map[timestep]

        # Load VTK file as a point cloud
        p = load(str(vtk_file))
        p.point_size(point_size)

        # Apply colormap based on 'activated' data if available
        if "activated" in p.pointdata.keys():
            p.cmap(lut, p.pointdata["activated"])
        else:
            p.c("blue") # Default color if 'activated' data is not present

        plt.clear() # Clear previous frame content

        # Add a text overlay for the current timestep
        timestep_text = Text2D(
            f"Timestep: {timestep}",
            pos="top-right",
            s=1.8,
            c="black",
            bg="white",
            alpha=0.8,
            font="Calco",
        )

        # Show the current frame with or without axes
        if show_axes:
            plt.show(p, axes, timestep_text, interactive=False, resetcam=False)
        else:
            plt.show(p, timestep_text, interactive=False, resetcam=False)

        video.add_frame() # Add the rendered frame to the video

        # Print progress
        progress = 100 * (frame_num + 1) / total_frames
        print(
            f"\rProcessing frame {frame_num + 1}/{total_frames} ({progress:.1f}%)",
            end="",
            flush=True,
        )

        del p # Release memory used by the point cloud

    print()
    video.close() # Finalize and save the video
    plt.close() # Close the plotter

    elapsed_time = time.time() - start_time
    print(f"Video creation time: {elapsed_time:.2f}s")
    print(f"Video saved to: {video_path.resolve()}")

    return video_path

# Example usage:
if __name__ == '__main__':
    # Ensure the environment is set up (clone, cd, install vedo, compile and run foundation)
    # This part would typically be handled by a shell script or manual setup before running this Python script.
    # For direct execution, you'd need to ensure the 'output' directory with VTK files exists.
    
    # You can uncomment and adapt these if you want to run the full setup from Python as well,
    # but generally, it's better to separate the simulation/data generation from the visualization script.
    # if os.path.exists('yalla_basic'):
    #     os.system('rm -rf yalla_basic')
    # os.system('git clone https://github.com/faebbs/yalla_basic.git')
    # os.chdir('yalla_basic/yalla-main')
    # os.system('pip install vedo')
    # os.system('nvcc -std=c++11 -arch=sm_75 model.cu -o model')
    # os.system('./model')

    # Call the video creation function
    create_video(
        output_dir="output",
        file_prefix="file_",
        start_timestep=0,
        end_timestep=1000,
        video_name="simulation",
        skip_frames=10,
        fps=2,
        show_axes=False,
    )