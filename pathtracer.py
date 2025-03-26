import subprocess
from pathlib import Path
import json
import imageio.v2 as iio
import os
import time

############# CLASS DEFINITION #############

class Pathtracer:
    """
    Pathtracer: a simple wrapper to manage the pathtracer project. This
    class contains docstrings. Additionally, it can be run as a script
    and generate a video.
    """

    def __init__(self, project_path=None):
        """
        Initialization requires only the following, assuming that the
        project directory is intact and not reorganized:

        Inputs:
            project_path (str), optional: Absolute path to Pathtracer root directory.

        Outputs:
            None
        """
        # List of paths to be used in the wrapper
        if project_path == None:
            self.project_path = Path(__file__).parent
        else:
            self.project_path = Path(project_path)
        self.config_path = self.project_path / Path("config.json")
        self.bin_path = self.project_path / Path("bin/main")

        # We need to load config.json into memory as a dict
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)

    def set_config(self):
        """
        Sets the config.json file in the Pathtracer directory to
        the current value of the config dictionary stored in the
        class.

        To edit a config setting, it should look something like this:

            Pathtracer.config["camera"]["rotation"] = [0, 1, 0]
            Pathtracer.set_config()

        Inputs:
            None
        Outputs:
            None
        """
        #take dict and save to file
        with open(self.config_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def call(self, output_path=None):
        """
        Calls the Pathtracer executable.

        Inputs:
            output_path (str), optional: Absolute path to output file of executable.
        Outputs:
            None
        """
        if output_path == None:
            path = self.project_path / Path("main.png")
        else:
            path = Path(output_path)

        args = [str(self.bin_path), str(path)]
        completed_process = subprocess.run(args, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)

        if completed_process.returncode != 0:
            raise Exception(f"Program call failed with return code {completed_process.returncode}")
        
    def animate(self, update_func, param_list, temp_dir_path=None, save_gif_path=None, duration=100):
        """
        Creates a .gif by repeatedly calling Pathtracer.call() with different settings.
        It requires a function, update_func, which is invoked every time a new image is
        generated which should update the config dictionary. It also requires a list of
        parameters, param_list, which will be iterated over and used to update the config
        dictionary. The function also prints the current progress to the terminal.

        Here is an example program:

            # initialize list of parameters to adjust
            yaw_list = [0.015*i for i in range(-3,4)]
            param_list = [ [0,yaw,0] for yaw in yaw_list]

            # needs an update function
            def update(param):
                tracer.config["camera"]["rotation"] = param

            tracer.animate(update, param_list)

        Inputs:
            update_func(param) (function): A function accepting a parameter which updates
                the config dictionary every frame.
            param_list (list): A list of the parameters iterated over in the .gif. Every
                frame, update_func operates on one entry from param_list.
        Outputs:
            temp_dir_path (str): A path to a temporary directory where images are created,
                stored, and later deleted when the function call ends.
            save_gif_path (str): A path to the location that the .gif will be saved.
            duration (float): Time between frames in ms.
        """
        # figuring out paths to save stuff to 
        if temp_dir_path == None:
            temp_dir = self.project_path/Path("temp_images")
        else:
            temp_dir = temp_dir_path

        if save_gif_path == None:
            path = self.project_path / Path("main.gif")
        else:
            path = save_gif_path

        try:
            # Create directory to store images in
            os.mkdir(temp_dir)

            # filenames to save temp images as
            filenames = [temp_dir/Path(f"temp{i}.png") for i in range(0, len(param_list))]
            
            # fundamental loop of the program:
            index = 0
            time_start = time.time()
            time_remaining = 60*60 
            for param in param_list:
                if index == 0:
                    print(f"Calculating image {index} of {len(param_list)}. Estimated time remaining (min:sec): ??:??", end='\r')
                else:
                    print(f"Calculating image {index} of {len(param_list)}. Estimated time remaining (min:sec): {int(time_remaining/60):02d}:{round(time_remaining%60):02d}", end='\r')

                # update config file
                update_func(param)
                #self.config["camera"]["rotation"] = param
                self.set_config()

                # run program and save to temp directory
                self.call(filenames[index])

                #update index and time
                time_end = time.time()
                time_elapsed = time_end - time_start
                time_start = time.time()
                
                index += 1
                time_remaining = time_elapsed*(len(param_list) - index)

            # save images as gif
            with iio.get_writer(path, mode='I', duration=duration, loop=0) as writer:
                for filename in filenames:
                    image = iio.imread(filename)
                    writer.append_data(image)
        
        finally:
            # clean up files and directories
            try: 
                for filename in filenames:
                    os.remove(filename)
            finally:
                os.removedirs(temp_dir)
                print('\n', end='\r')
        


############# RUN IF EXECUTED AS MAIN #############

if __name__ == "__main__":

    tracer = Pathtracer()

    # initialize list of parameters to adjust
    yaw_list = [0.015*i for i in range(-3,4)]
    param_list = [ [0,yaw,0] for yaw in yaw_list]

    # needs an update function
    def update(param):
        tracer.config["camera"]["rotation"] = param

    tracer.animate(update, param_list)