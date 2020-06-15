from config import script_dir

from IPython.display import Audio

sound_file = script_dir + 'beep.wav'

Audio(sound_file, autoplay=True)



