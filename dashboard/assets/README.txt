Place your FIXED target image here as:

    target.png

This is the single preloaded target that every uploaded photo is aligned toward.
It is loaded once at server startup and kept on the GPU.

To use a different file name or location, set the TARGET_PATH environment variable:

    TARGET_PATH=/path/to/my_target.jpg python dashboard/app.py
