import subprocess
import os
""" 
AmpSlicerl, OnsetSlicer are Flucoma slicers from the CLI toolset. 
AmpSlicer slices based on amplitude shifts in the spectral representation
of the sound. 
OnsetSlicer _

"""


class OnsetSlicer:
    EXE = "fluid-onsetslice"

    def __init__(
        self,
        # fftsettings=[1024, -1, -1],
        # filtersize=5,
        # framedelta=0,
        # maxfftsize=-1,
        metric=9, #phase dev
        # minslicelength=2,
        # numchans=-1,
        # numframes=-1,
        # startchan=0,
        # startframe=0,
        threshold=0.2
        # warnings=0
    ):
        # self.fftsettings = fftsettings
        # self.filtersize = filtersize
        # self.framedelta = framedelta
        # self.indices = indices
        # self.maxfftsize = maxfftsize
        # self.minslicelength = minslicelength
        # self.numchans = numchans
        # self.numframes = numframes
        self.threshold = threshold
        self.metric = metric
        self.name = self.__class__.__name__

    def run(self, source, output_file_name) -> None:
        """
        Constructs and executes the fluid-onsetslice command.
        source: is the wav file 
        output_file_name: csv file name
        """
        output_folder = os.path.dirname(source) + "\\slicing\\" 
        if os.path.exists(output_file_name):
            output_path = os.path.join(output_folder, output_file_name)
        
        command = [
            "fluid-onsetslice",
            "-source", source,
            "-indices", output_path,
            "-metric", str(self.metric), # rectified phase dev. How much the next spectral image differs from the anticipated prior
            "-threshold", str(self.threshold)
        ]
        try:
            result = subprocess.run(
                command, 
                check=True,          
                capture_output=True, 
                text=True            
            )
        except subprocess.CalledProcessError as e:
            print("Error details:", e.stderr)





class AmpSlicer:
    EXE = "fluid-ampslice"
    def __init__(
        self,
        fastrampdown=1,
        fastrampup=1,
        floor=-144.0,
        highpassfreq=85.0,
        minslicelength=2,
        numchans=-1,
        numframes=-1,
        offthreshold=-144.0,
        onthreshold=144.0,
        slowrampdown=100,
        slowrampup=100,
        startchan=0,
        startframe=0,
        warnings=0
    ):
        self.fastrampdown = fastrampdown
        self.fastrampup = fastrampup
        self.floor = floor
        self.highpassfreq = highpassfreq
        self.minslicelength = minslicelength
        self.numchans = numchans
        self.numframes = numframes
        self.offthreshold = offthreshold
        self.onthreshold = onthreshold
        self.slowrampdown = slowrampdown
        self.slowrampup = slowrampup
        self.startchan = startchan
        self.startframe = startframe
        # self.warnings = warnings

        self.name = self.__class__.__name__

    
    def run(self, output, source):
        """ 
        Output should be a csv file. Source is the (Corpus object)
        """
        print(source.path)
        command = [
            self.EXE,
            "-source", str(source.path),
            "-indices", str(output), 
            "-fastrampdown", str(self.fastrampdown),
            "-fastrampup", str(self.fastrampup),
            "-floor", str(self.floor),
            "-highpassfreq", str(self.highpassfreq),
            "-minslicelength", str(self.minslicelength),
            "-numchans", str(self.numchans),
            "-numframes", str(self.numframes),
            "-offthreshold", str(self.offthreshold),
            "-onthreshold", str(self.onthreshold),
            "-slowrampdown", str(self.slowrampdown),
            "-slowrampup", str(self.slowrampup),
            "-startchan", str(self.startchan),
            "-startframe", str(self.startframe)
        ]
        try:
            result = subprocess.run(
                command, 
                check=True,          # Raises CalledProcessError if the exit code is non-zero
                capture_output=True, # Captures stdout and stderr
                text=True            # Returns output as strings instead of bytes
            )
            print("Command Output:\n", result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running fluid-noveltyslice: {e}")
            print("Error details:", e.stderr)

        source.slices[source.name] = source.slices.get(source.name, []).append({output: self})

