import os # For changing directories and handling paths
import subprocess # For running shell commands (compilation and execution)
from pathlib import Path # For handling file paths in a platform-independent way
import shutil # For copying files and directories
import time # For measuring execution time
import math # For any mathematical operations needed in the future
import sys # For handling command-line arguments
import vedo # For visualization and video creation from VTK files
import glob, os, re # For file handling and regular expressions, if needed for filename parsing
import imageio # For reading/writing video files
import numpy # For numerical operations, if needed for processing VTK data
# --- vedo imports ---
from vedo import *

def compile_and_run_cuda(cuda_file_path: Path, output_executable_path: Path):
    """
    Compiles a CUDA .cu file and runs the resulting executable.
    The executable is expected to generate its output in a directory named 'output'
    within the current working directory of its execution.
    Args:
        cuda_file_path (Path): Path to the CUDA source file (.cu).
        output_executable_path (Path): Desired path for the compiled executable.
    Raises:
        subprocess.CalledProcessError: If compilation or execution fails, the error details 
        will be printed and the exception will be raised.
    """
    print(f"\n--- Compiling {cuda_file_path.name} ---")
    compile_command = [
        "nvcc",
        "-std=c++11",
        "-arch=sm_75", # Using sm_75 as per previous successful compilation
        str(cuda_file_path), 
        "-o",
        str(output_executable_path)
    ] # Compile command for nvcc, specifying C++11 standard and architecture. Adjust as needed for different CUDA versions or architectures.

    try:
        compile_result = subprocess.run(compile_command, check=True, capture_output=True, text=True) # Run the compile command, capturing stdout and stderr. check=True will raise an error if compilation fails.
        print(f"Compilation successful for {cuda_file_path.name}") # Print success message if compilation succeeds.
        if compile_result.stdout: print(f"STDOUT:\n{compile_result.stdout}") # Print any standard output from the compilation process, if present.
        if compile_result.stderr: print(f"STDERR:\n{compile_result.stderr}") # Print any standard error output from the compilation process, if present. This can include warnings or errors from the compiler.

    except subprocess.CalledProcessError as e: # If compilation fails, this block will execute, printing the error details and re-raising the exception.
        print(f"Error compiling {cuda_file_path.name}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

    print(f"\n--- Executing {output_executable_path.name} ---") # Print message indication start of execution
    try: 
        pass
    except subprocess.CalledProcessError as e: # If execution fails, this block will execute, printing the error details and re-raising the exception.
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
    skip_frames=1,
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
        point_size (int): Size of the points in the visualization (unused in sphere mode,
                          kept for API compatibility).
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

   # ── Camera calibration based on the first frame ──────────────────────────
    # We read the positions from the first VTK file to automatically calculate
    # the center and the required camera distance. This makes the camera work
    # independently of how large the cylinder is.
    first_vtk = timestep_map[selected_timesteps[0]]
    first_pts = load(str(first_vtk))
    pos = first_pts.points             # numpy array (N, 3): all cell positions


    # Center of the cylinder in x and y (cross-sectional plane)
    cx = (pos[:, 0].max() + pos[:, 0].min()) / 2
    cy = (pos[:, 1].max() + pos[:, 1].min()) / 2


    # Center along the z-axis (longitudinal axis of the cylinder)
    cz = (pos[:, 2].max() + pos[:, 2].min()) / 2


    # Extent of the cylinder along z (= length)
    z_spread = pos[:, 2].max() - pos[:, 2].min()


    # Radius of the cylinder (extent in x or y)
    xy_spread = max(
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
    )


    # The camera is positioned far along the +x-axis and looks along the x-direction
    # at the longitudinal axis of the cylinder (the z-axis then runs horizontally in the image).
    # cam_x_distance: large enough so that the entire cylinder fits into the image.
    cam_x_distance = max(z_spread, xy_spread) * 4


    print(f"Cylinder center:  x={cx:.3f}, y={cy:.3f}, z={cz:.3f}")
    print(f"Cylinder extents: z_spread={z_spread:.3f}, xy_spread={xy_spread:.3f}")
    print(f"Camera distance:  {cam_x_distance:.3f}")


    # ── Initialize plotter ──────────────────────────────────────────────────
    plt = Plotter(size=(2000, 2000), bg="white", offscreen=True)
    plt.renderer.SetBackground(1, 1, 1)  # White background (RGB 0–1)


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


    # --- Removed color lookup table (lut) as it's no longer used for coloring ---


    video = Video(str(video_path), fps=fps, backend="ffmpeg")


    total_frames = len(selected_timesteps)
    print(f"Found VTK files: {len(vtk_files)}")
    print(f"Used frames:     {total_frames}")
    print(f"Video output:    {video_path.resolve()}")
    print()


    for frame_num, timestep in enumerate(selected_timesteps):
        vtk_file = timestep_map[timestep]


        p = load(str(vtk_file))
        positions = p.points           # numpy array (N, 3)


        # ── Render cells as spheres ─────────────────────────────────────────
        # Spheres() instead of point_size: each cell is drawn as a shiny 3D sphere.
        # r=0.25 is a good starting value for r_min=0.5 (half the equilibrium
        # distance → spheres almost touch each other).
        spheres = Spheres(
            positions,
            r=0.25,   # Sphere radius – adjust if needed (r_min / 2 recommended)
            res=12,   # Polygon resolution per sphere (higher = smoother, but slower)
            c="lightblue", # Always set to lightblue as requested
        )


        # --- Removed conditional coloring based on 'neighbours' or 'activated' properties ---
        spheres.lighting("glossy")       # Shiny sphere look (Phong lighting)


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


        # ── Camera: side view of the cylinder ───────────────────────────
        # The camera is placed on the +x-axis and looks in the -x direction at the cylinder.
        # The z-axis (longitudinal axis) then runs horizontally in the image:
        # left z=0, right z=z_max → the pulse wave becomes visible from left to right.
        camera_settings = {
            "pos":        [cx + cam_x_distance, cy, cz],  # Camera to the right on the x-axis
            "focalPoint": [cx,                  cy, cz],  # Focus point: cylinder center
            "viewup":     [0, 1, 0],                       # y is "up" in the image
        }


        if show_axes:
            plt.show(
                spheres, axes, timestep_text,
                resetcam=(frame_num == 0),
                camera=camera_settings,
                interactive=False,
            )
        else:
            plt.show(
                spheres, timestep_text,
                resetcam=(frame_num == 0),  # Re-center only for the first frame
                camera=camera_settings,
                interactive=False,
            )


        video.add_frame()

        progress = 100 * (frame_num + 1) / total_frames
        print(
            f"\rProcessing frame {frame_num + 1}/{total_frames} ({progress:.1f}%)",
            end="",
            flush=True,
        )

        del p, spheres

    print()
    video.close()
    plt.close()

    elapsed_time = time.time() - start_time
    print(f"Video creation time: {elapsed_time:.2f}s")
    print(f"Video saved to: {video_path.resolve()}")

    return video_path

def main():
    '''
    Main function to compile and run a CUDA file, then create a video from its output.
    The script expects a CUDA filename as a command-line argument and processes it accordingly.

    Raises:      SystemExit: If no command-line argument is provided for the CUDA filename, the function will print usage instructions and exit the program.
                 subprocess.CalledProcessError: If there is an error during compilation or execution of the CUDA file, the error details will be printed and the exception will be raised.
                 FileNotFoundError: If the expected output directory is not found after execution, a warning will be printed and video creation will be skipped for that run.
    Notes:       - The script assumes it is run from the 'analyse' subdirectory of the repository and that the repository is already cloned.
                 - The CUDA file should generate its output in a directory named 'output' within the current working directory of its execution for the video creation to work correctly.
                 - The script creates a new numbered directory for each run within a base 'yalla_runs' directory to store the outputs and videos, ensuring that previous runs are not overwritten.  
    '''
    # Check for command-line argument for the CUDA filename
    if len(sys.argv) < 2:
        print("Usage: python script.py <cuda_filename.cu>")
        print("Example: python script.py springs.cu")
        sys.exit(1)

    cuda_file_name = sys.argv[1] 

    # Assume the script is run from the yallamain/analyse directory
    yalla_dir = Path.cwd()

    # Define a dedicated base directory for all numbered outputs within yalla_dir
    base_output_runs_dir = yalla_dir / "yalla_runs"
    base_output_runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created base run directory for outputs: {base_output_runs_dir}")

    # The repository is assumed to be already cloned and the script executed from its analyse subdirectory.

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
            print(f"EXECUTABLE STDERR:\n{e.stderr}") # stderr = error messages from the executable

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
