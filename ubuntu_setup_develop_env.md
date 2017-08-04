### Ubuntu 16.04

** edit .bashrc **
if [ "$color_prompt" = yes ]; then
    #PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
    ** PS1='\[\033[01;34m\]\W\[\033[00m\]\$ ' **
else
    # PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
    ** PS1='\W\$ ' **
fi

#### Before doing any installations
```
sudo apt-get update
```

#### 倉頡輸入法
```
sudo apt-get install fcitx-table-cangjie3
```
#### Python3
```
sudo apt-get install -y python3-pip python3-setuptools
sudo -H pip3 install --upgrade pip numpy ipykernel jupyter matplotlib Pillow
```

#### Tensorflow
[reference](https://www.tensorflow.org/install/install_linux)
```
sudo apt-get install libcupti-dev
sudo apt-get install python3-pip python3-dev
sudo -H pip3 install tensorflow
```
Install GPU support version ok but unable to run. ?????
sudo -H pip3 install tensorflow-gpu # Python 3.n; GPU support.

#### Git
```
sudo apt-get install git
```
** edit ~/.gitconfig **
```
[core]
	editor = vi

[user]
	name = johnnycc_huang
	email = johnnycc_huang@compal.com

[credential "https://github.com"]
	username = johnny610926

[color]
	diff = auto
	status = auto
	branch = auto
	log = auto

[push]
	default = simple

[merge]
	tool = meld
```

#### Support exfat filesystem type

```
sudo apt-get install exfat-utils exfat-fuse
```

#### SSH server (ssh telnet)
```
sudo apt-get install openssh-server
```

#### Chrome
download Chrome desktop
```
sudo dpkg -i google-chrome-stable_current_amd64.deb
```
Login google account

#### Visual Studio Code
download Visual Studio Code
```
sudo dpkg -i code_1.14.2-1500506907_amd64.deb
```


