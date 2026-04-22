import os
import subprocess
from pathlib import Path
import shutil
import time
import math
import sys 

# --- vedo imports ---
from vedo import *

def compile_and_run_cuda(cuda_file_path: Path, output_executable_path: Path):
    """
    Compiles a CUDA .cu file and runs the resulting executable.
    The executable is expected to generate its output in a directory named 'output'
    within the current working directory of its execution.
    """
    print(f"\n--- Compiling {cuda_file_path.name} ---")
    compile_command = [
        "nvcc",
        "-std=c++11",
        "-arch=sm_75", # Using sm_75 as per previous successful compilation
        str(cuda_file_path),
        "-o",
        str(output_executable_path)
    ]
    try:
        compile_result = subprocess.run(compile_command, check=True, capture_output=True, text=True)
        print(f"Compilation successful for {cuda_file_path.name}")
        if compile_result.stdout: print(f"STDOUT:\n{compile_result.stdout}")
        if compile_result.stderr: print(f"STDERR:\n{compile_result.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error compiling {cuda_file_path.name}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

    print(f"\n--- Executing {output_executable_path.name} ---")
    try:
        pass
    except subprocess.CalledProcessError as e: 
        print(f"Error executing {output_executable_path.name}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

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

    Args:
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

    Raises:
        FileNotFoundError: If the output directory or no VTK files are found.
        ValueError: If no timesteps can be extracted from filenames or no files
                    are found within the selected timestep range.
    """
    start_time = time.time()

    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path.resolve()}")

    vtk_files = sorted(output_path.glob(f"{file_prefix}*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(
            f"No VTK files found in {output_path.resolve()} with prefix '{file_prefix}'"
        )

    def extract_timestep(path):
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
        raise ValueError("Could not read timesteps from filenames.")

    available_timesteps = sorted(timestep_map.keys())

    if end_timestep is None:
        end_timestep = available_timesteps[-1]

    selected_timesteps = [
        ts for ts in available_timesteps
        if start_timestep <= ts <= end_timestep
    ][::skip_frames]

    if not selected_timesteps:
        raise ValueError("No files found in the selected timestep range.")

    video_path = output_path / f"{video_name}.mp4"

    plt = Plotter(size=(2000, 2000), bg="white", offscreen=True)

    plt.camera.SetPosition(0, 0, 30)
    plt.camera.SetFocalPoint(4, 0, 0)
    plt.camera.SetViewUp(0, 1, 0)

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

    video = Video(str(video_path), fps=fps, backend="ffmpeg")

    total_frames = len(selected_timesteps)
    print(f"Found VTK files: {len(vtk_files)}")
    print(f"Used frames: {total_frames}")
    print(f"Video-Output: {video_path.resolve()}")
    print()

    for frame_num, timestep in enumerate(selected_timesteps):
        vtk_file = timestep_map[timestep]

        p = load(str(vtk_file))
        p.point_size(point_size)

        if "activated" in p.pointdata.keys():
            p.cmap(lut, p.pointdata["activated"])
        else:
            p.c("blue")

        plt.clear()

        timestep_text = Text2D(
            f"Timestep: {timestep}",
            pos="top-right",
            s=1.8,
            c="black",
            bg="white",
            alpha=0.8,
            font="Calco",
        )

        if show_axes:
            plt.show(p, axes, timestep_text, interactive=False, resetcam=False)
        else:
            plt.show(p, timestep_text, interactive=False, resetcam=False)

        video.add_frame()

        progress = 100 * (frame_num + 1) / total_frames
        print(
            f"\rProcessing frame {frame_num + 1}/{total_frames} ({progress:.1f}%)",
            end="",
            flush=True,
        )

        del p

    print()
    video.close()
    plt.close()

    elapsed_time = time.time() - start_time
    print(f"Video creation time: {elapsed_time:.2f}s")
    print(f"Video saved to: {video_path.resolve()}")

    return video_path

def main():
    # Check for command-line argument for the CUDA filename
    if len(sys.argv) < 2:
        print("Usage: python script.py <cuda_filename.cu>")
        print("Example: python script.py springs.cu")
        sys.exit(1)

    cuda_file_name = sys.argv[1]

    # Assume the script is run from the yalla_basic/yalla-main directory
    yalla_dir = Path.cwd()

    # Define a dedicated base directory for all numbered outputs within yalla_dir
    base_output_runs_dir = yalla_dir / "yalla_runs"
    base_output_runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created base run directory for outputs: {base_output_runs_dir}")

    # The repository is assumed to be already cloned and the script executed from its yalla-main subdirectory.
    # No need to clone or check for vedo installation here.

    print(f"\n{'='*50}")
    print(f"Processing example: {cuda_file_name}")
    print(f"{'='*50}")

    # Create a new numbered directory for this specific run
    # Increment output_run_counter based on existing runs
    existing_runs = sorted(base_output_runs_dir.glob("run_*"))
    output_run_counter = 0
    if existing_runs:
        last_run_num = int(existing_runs[-1].name.split('_')[1])
        output_run_counter = last_run_num + 1

    current_run_dir = base_output_runs_dir / f"run_{output_run_counter:03d}"
    current_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run directory for outputs: {current_run_dir}")

    cuda_source_path = yalla_dir / "analyse" / cuda_file_name
    executable_name = cuda_file_name.replace(".cu", "")
    executable_path_in_run_dir = current_run_dir / executable_name

    # Initialize original_cwd before the try block to prevent UnboundLocalError
    original_cwd = Path.cwd()

    try:
        # Compile the CUDA file
        compile_and_run_cuda(cuda_source_path, executable_path_in_run_dir)

        # Temporarily change CWD for executing the compiled program.
        # This ensures the 'output' directory created by the program is inside current_run_dir.
        os.chdir(current_run_dir)
        print(f"Changed CWD to: {Path.cwd()} for execution.")

        # Run the compiled program, explicitly using './' for execution in current directory
        execute_result = subprocess.run([f"./{executable_name}"], check=True, capture_output=True, text=True)
        print(f"Executed {executable_name} in {current_run_dir}")
        if execute_result.stdout:
            print(f"EXECUTABLE STDOUT:\n{execute_result.stdout}") # stdout = output from the executable
        if execute_result.stderr:
            print(f"EXECUTABLE STDERR:\n{execute_result.stderr}") # stderr = error messages from the executable

        # The CUDA program will have created its 'output' directory inside current_run_dir
        actual_output_dir_for_video = current_run_dir / "output"

        # Restore original CWD
        os.chdir(original_cwd)
        print(f"Restored CWD to: {original_cwd}")

        # Verify the output directory and create video
        if not actual_output_dir_for_video.exists():
            print(f"Warning: '{actual_output_dir_for_video}' not found after running {executable_name}. Skipping video creation for this run.")
        else:
            print(f"Contents of {actual_output_dir_for_video}:")
            for item in actual_output_dir_for_video.iterdir():
                print(f"  - {item.name}")
            print()
            print(f"Found program's output directory: {actual_output_dir_for_video}")
            video_name_prefix = f"{executable_name}_run_{output_run_counter:03d}"
            created_video_path = create_video(
                output_dir=str(actual_output_dir_for_video),
                file_prefix=f"{executable_name}_", # Changed prefix from "file_" to match executable output
                start_timestep=0,
                end_timestep=1000,
                video_name=video_name_prefix,
                skip_frames=10,
                fps=2,
                show_axes=False,
            )
            print(f"Video created at: {created_video_path}")

    except Exception as e:
        print(f"An error occurred during processing {cuda_file_name} in run_{output_run_counter:03d}: {e}")
        # Ensure CWD is restored even on error during execution
        if Path.cwd() != original_cwd:
            os.chdir(original_cwd)
            print(f"Restored CWD to: {original_cwd} after error.")

    print(f"{'='*50}\n")
    print("Script finished processing.")


if __name__ == "__main__":
    main()