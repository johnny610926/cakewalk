### macOS

- Change X-terminal color and shorten command line prompt
    Edit ~/.bash_profile and add the following code.
    ```
    export CLICOLOR=1
    export LSCOLORS=gxbxcxdxexegedabagaced
    export PS1='\[\033[01;34m\]\W\[\033[00m\]\$ '
    ```

- Homebrew
    [Homebrew](https://brew.sh/) is the missing package manager for macOS. It can install the stuff you need that Apple didn’t.
    ```
    $ sudo ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    ```

- Pythone
    * [Pyenv](https://github.com/pyenv/pyenv)
        Python Version Management Toolkit
        ```
        $ brew install pyenv
        $ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
        $ exec $SHELL
        ```
    * Python3.5
        ```
        $ brew install zlib
        $ xcode-select --install
        $ pyenv install 3.5.3
        $ pyenv global 3.5.3
        ```
    * Python modules
        ```
        pip3 install --upgrade pip numpy scipy sklearn ipykernel jupyter matplotlib Pillow pandas scikit-learn
        ```

- Tensorflow
    ```
    $ pip install --upgrade https://bazel.blob.core.windows.net/tensorflow/tensorflow-1.2.1-cp35-cp35m-macosx_10_12_x86_64.whl
    ```

- OpenCV Python binding
    * Install video codec libraries
        ```
        $ brew install ffmpeg webp
        ```
    * Install OpenCV
        ```
        pip install https://bazel.blob.core.windows.net/opencv/opencv_python-3.2.0-cp35-cp35m-macosx_10_12_x86_64.whl
        ```

- Git
    ```
    brew install git
    ```
    Edit ~/.bash_profile and add the following code
    ```
    export PATH=/usr/local/bin:$PATH
    ```
    Check the version
    ```
    git --version
    ```
